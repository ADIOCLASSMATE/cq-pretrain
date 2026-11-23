"""Real-data input for training: per-epoch loaders and DP-agreed microbatch groups."""

import json
import os
import time

import torch
import torch.distributed as dist

import distributed.accel as accel
import distributed.megatron as P

ACCEL = accel.accel_type()
DEV = accel.device_module()

# Per-step counters summed over the DP group. Order is irrelevant to the sum;
# names keep every consumer independent of positions.
COUNT_FIELDS = (
    "supervised_text_tokens",
    "images_consumed",
    "body_tokens",
    "occupied_tokens",
    "active_microbatches",
    "generation_tokens",
    "documents_completed",
    "t2i_captions_completed",
    "i2t_captions_completed",
    "backbone_processed_slots",
    "padding_tokens",
    "dit_microbatches",
    "generation_images",
    "condition_query_tokens",
    "unconditional_images",
    "instruction_dropped_images",
    "instruction_dropout_eligible_images",
)


def _local_counts(micro_batches):
    packs = [x for x in micro_batches if x]
    documents = [d for _, receipt in packs for d in receipt["documents"]]
    return dict(
        supervised_text_tokens=sum(b.text_indices.numel() for b, _ in packs),
        images_consumed=sum(len(b.image_indices) for b, _ in packs),
        body_tokens=sum(r["body_tokens"] for _, r in packs),
        occupied_tokens=sum(r["occupied"] for _, r in packs),
        active_microbatches=len(packs),
        generation_tokens=sum(r.get("generation_tokens", 0) for _, r in packs),
        documents_completed=sum(d["completed"] for d in documents),
        t2i_captions_completed=sum(d["completed"] and d.get("direction") == "T2I" for d in documents),
        i2t_captions_completed=sum(d["completed"] and d.get("direction") == "I2T" for d in documents),
        backbone_processed_slots=sum(b.ids.shape[1] + len(b.mainline.query_positions) for b, _ in packs),
        padding_tokens=sum(b.ids.shape[1] - r["occupied"] for b, r in packs),
        dit_microbatches=sum(r.get("dit_microbatches", int(bool(b.generation_image_count))) for b, r in packs),
        generation_images=sum(b.generation_image_count for b, _ in packs),
        condition_query_tokens=sum(r["condition_query_tokens"] for _, r in packs),
        unconditional_images=sum(r["unconditional_images"] for _, r in packs),
        instruction_dropped_images=sum(r["instruction_dropped_images"] for _, r in packs),
        instruction_dropout_eligible_images=sum(r["instruction_dropout_eligible_images"] for _, r in packs),
    )


class DataPipeline:
    """Owns the document stream, packer and device loader of the current epoch."""

    def __init__(self, args, cfg, data, resumed):
        from data.codec import DevicePreparer
        from data.runtime import empty_batch

        self.args, self.cfg, self.tokens = args, cfg, data.tokens
        self.device = f"{ACCEL}:{int(os.environ.get('LOCAL_RANK', 0))}"
        # Spawned CPU workers must not autoload the accelerator backend on import.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        with torch.random.fork_rng(devices=[DEV.current_device()], device_type=ACCEL):
            self.preparer = DevicePreparer(
                cfg,
                device=self.device,
                vae_dtype=getattr(torch, args.vae_dtype),
                encoder=args.latent_encoder if args.data_mode == "A" else None,
            )
        self.epoch = resumed["documents"]["signature"]["epoch"] if resumed else 0
        if not 0 <= self.epoch < args.data_epochs:
            raise ValueError("Invalid saved data epoch")
        self._open(self.epoch, resumed)
        self.padding_batch = empty_batch(cfg, data.tokens).to(ACCEL)

    def _open(self, epoch, cursor=None):
        from data.device import DeviceLoader
        from data.packing import ConsumedLoader, Packer
        from data.stream import DocumentStream

        args = self.args
        self.docs = DocumentStream(
            args.data_root,
            args.data_mode,
            P.dp_rank(),
            P.dp_size(),
            args.seed,
            epoch=epoch,
            workers=args.data_workers,
            prefetch=args.data_prefetch,
            source_root=args.source_root,
            block_rows=args.shuffle_block_rows,
            tokenize_rows=args.tokenize_rows,
            image_threads=args.image_workers,
            image_prefetch=args.image_prefetch,
            state=cursor["documents"] if cursor else None,
            allow_development=args.allow_development_data,
            cfg=self.cfg,
            sequence_length=args.sequence_length,
        )
        self.packer = Packer(
            self.docs,
            self.cfg,
            args.sequence_length,
            self.tokens,
            cursor or None,
            pin_device=self.device,
            max_images=args.max_images_per_pack,
            padding_multiple=args.pack_padding_multiple,
        )
        host = ConsumedLoader(self.packer, args.data_prefetch, cursor or None)
        self.loader = DeviceLoader(host, self.preparer, args.device_prefetch)

    def next_group(self, rank, completed_step, output, on_epoch_end):
        """One accumulation group and the DP-summed counts.

        Every DP rank leaves together: a rank whose epoch is exhausted rolls
        over only when all ranks are. `on_epoch_end` runs before the exhausted
        loader closes, so a checkpoint there records the final cursor.
        Returns (micro_batches, counts, wait_seconds); micro_batches is None
        once the last epoch has no data on any rank.
        """
        begin = time.perf_counter()
        while True:
            micro_batches = self.loader.next_group(self.args.grad_accum)
            local = _local_counts(micro_batches)
            values = torch.tensor([local[k] for k in COUNT_FIELDS], device=ACCEL, dtype=torch.int64)
            dist.all_reduce(values, group=P.state().get_data_parallel_group())
            counts = dict(zip(COUNT_FIELDS, values.tolist(), strict=True))
            if counts["active_microbatches"]:
                break
            if self.args.data_epochs > 1:
                self._record_epoch_end(rank, completed_step, output)
            if self.epoch + 1 == self.args.data_epochs:
                break
            on_epoch_end()
            self.loader.close()
            self.epoch += 1
            self._open(self.epoch)
        wait = time.perf_counter() - begin
        if not counts["active_microbatches"]:
            return None, counts, wait
        # Each DP replica may have a different number of packs. Its final real
        # microbatch launches the same ordered DP reduction sequence. Only an
        # exhausted replica needs one zero-loss graph.
        return [x for x in micro_batches if x] or [None], counts, wait

    def _record_epoch_end(self, rank, completed_step, output):
        from utils.checkpoint_catalog import durable_json

        cursor = self.loader.state_dict()
        at_end = cursor["documents"]["position"] == {"block": self.docs.total, "row": 0}
        if not at_end or cursor["element"] or cursor["offset"]:
            raise RuntimeError("Epoch ended before the last complete document was acknowledged")
        durable_json(
            output / f"epoch_{self.epoch:02d}_rank_{rank:05d}.json",
            dict(
                epoch=self.epoch,
                step=completed_step,
                rank=rank,
                packs=cursor["serial"],
                documents=cursor["documents"],
            ),
        )

    def acquire(self, item):
        return self.loader.acquire(item)

    def acknowledge(self, receipt):
        self.loader.acknowledge(receipt)

    def device_wait_seconds(self):
        return self.loader.device_wait_seconds()

    def state_dict(self):
        return self.loader.state_dict()

    def close(self, output, rank):
        """Stop workers and write this rank's data statistics."""
        self.loader.close()
        image_stats = self.docs.image_reader.stats() if self.docs.image_reader else {}
        stats = dict(
            consumed_state=self.loader.state_dict(),
            prefetched_document_stats=self.docs.stats,
            packing_stats=self.packer.stats,
            preparer_stats=self.preparer.stats,
            device_stats=self.loader.stats,
            image_stats=image_stats,
        )
        (output / f"data_rank_{rank:05d}.json").write_text(json.dumps(stats))
