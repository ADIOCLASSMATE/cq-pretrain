"""Resumable extraction into the original task-neutral samples.jsonl."""

from copy import deepcopy
import json
import os
from pathlib import Path
import time
import torch
from .cache import (
    FORMATS_V2,
    branch_preprocessing,
    digest,
    encoder_contracts,
    load_cache,
    save_cache,
    tensor_info,
)
from .manifest import (
    atomic_write,
    file_lock,
    file_sha256,
    merge_latent_updates,
    original_digest,
    read_manifest,
    resolve_video,
)
from .video import prepare_video, prepare_qwen_inputs


def add_arguments(parser):
    parser.add_argument(
        "--in-place", action="store_true", help="Extend the existing JSONL atomically"
    )
    parser.add_argument("--profile-name", default="native81_fps2_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-ids", help="JSON array of sample IDs, used for a paired pilot"
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--verify-online",
        action="store_true",
        help="Compare actual online encodings and Dataset reads",
    )
    parser.add_argument(
        "--stop-after-files",
        type=int,
        help="Fault injection: stop after N newly saved files",
    )
    parser.add_argument("--run-id", default="extract")


class Encoders:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.qwen = self.wan = None

    @torch.no_grad()
    def encode(self, kind, rgb, pixels=None, grid=None):
        config = self.config
        if kind == "qwen":
            if self.qwen is None:
                from models.backbone_qwen35 import Qwen35VideoEncoder

                self.qwen = (
                    Qwen35VideoEncoder.from_pretrained(
                        config["model"]["qwen_path"],
                        dtype=torch.bfloat16
                        if config["training"]["mixed_precision"] == "bf16"
                        else torch.float32,
                        attn_implementation=config["model"]["attn_implementation"],
                    )
                    .to(self.device)
                    .eval()
                    .requires_grad_(False)
                )
            # Preserve the encoder's real output dtype, including FP32 experiments.
            return (
                self.qwen(pixels.to(self.device), grid[None].to(self.device))[0]
                .detach()
                .cpu()
            )
        if self.wan is None:
            from .codec import WanVideoCodec

            self.wan = WanVideoCodec.from_pretrained(
                config["codec"].get("path") or config["model"]["wan_path"],
                device=self.device,
                dtype=getattr(torch, config["codec"]["dtype"]),
            )
        return (
            self.wan.encode(rgb.permute(1, 0, 2, 3).float().div(127.5).sub(1))
            .detach()
            .cpu()
        )


def _validate_sequences(row, qwen, processor, config):
    from .sequence import make_video_sequence

    caption = row.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("In-place task-neutral extraction requires a caption")
    visual, text = (
        {"type": "video", "video_id": "v0"},
        {"type": "text", "text": caption},
    )
    lengths = {}
    for task, first, last in (("t2v", text, visual), ("v2t", visual, text)):
        content = (
            [first]
            if task == "t2v"
            else [
                first,
                {
                    "type": "text",
                    "text": row.get("instruction", "Describe this video."),
                },
            ]
        )
        sequence = make_video_sequence(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [last]},
            ],
            processor,
            qwen["grid_thw"],
            qwen["metadata"]["qwen_timestamps"],
            max_seq_length=config["data"]["max_seq_length"],
        )
        lengths[task] = len(sequence["input_ids"])
    return lengths


def _verify_dataset(manifest, sample_ids, processor, config, encoders):
    from .dataset import VideoDataset, read_records

    maximum = {"qwen": 0.0, "wan": 0.0}
    for task in ("t2v", "v2t"):
        source = {"path": str(manifest), "task": task}
        rows = [r for r in read_records(source) if r["sample_id"] in sample_ids]
        cached_config = deepcopy(config)
        cached_config["data"]["representations"] = {"qwen": "cache", "wan": "cache"}
        online_config = deepcopy(config)
        online_config["data"]["representations"] = {"qwen": "online", "wan": "online"}
        cached = VideoDataset(source, processor, cached_config, records=rows)
        online = VideoDataset(source, processor, online_config, records=rows)
        for i in range(len(rows)):
            a, b = cached[i], online[i]
            if a["clip"] != b["clip"]:
                raise ValueError("Online/cache clips differ")
            for key in ("input_ids", "video_grid_thw"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            # T2V and V2T share the same encoder inputs; encode each clip once here.
            if task == "t2v":
                qwen = encoders.encode(
                    "qwen", None, b["pixel_values_videos"], b["video_grid_thw"]
                )
                if qwen.dtype != a["video_features"].dtype:
                    raise ValueError(
                        "Qwen stored dtype differs from actual model output"
                    )
                torch.testing.assert_close(a["video_features"], qwen, rtol=0, atol=0)
                maximum["qwen"] = max(
                    maximum["qwen"],
                    float((a["video_features"].float() - qwen.float()).abs().max()),
                )
                wan = (
                    encoders.wan.encode(b["wan_rgb"]).detach().cpu()
                    if encoders.wan
                    else None
                )
                if wan is None:
                    # Load the encoder lazily on a pure verification restart.
                    rgb, _ = prepare_video(
                        rows[i]["videos"][0], config["data"]["video"]
                    )
                    wan = encoders.encode("wan", rgb)
                torch.testing.assert_close(a["wan_latent"], wan, rtol=1e-5, atol=1e-6)
                maximum["wan"] = max(
                    maximum["wan"], float((a["wan_latent"] - wan).abs().max())
                )
    return {
        "samples": len(sample_ids),
        "tasks": ["t2v", "v2t"],
        "max_abs_error": maximum,
    }


def extract_in_place(args):
    from configs.config import load_config
    from training.train import load_processor, register_device

    if args.output or not args.manifest:
        raise ValueError("--in-place requires --manifest and forbids --output")
    if (
        args.num_shards < 1
        or not 0 <= args.shard_index < args.num_shards
        or args.flush_every < 1
    ):
        raise ValueError("Invalid shard/flush parameters")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    for name in (args.profile_name, args.run_id):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Profile/run names must be single path components")
    manifest = Path(args.manifest).resolve()
    if manifest.suffix != ".jsonl":
        raise ValueError("In-place extraction requires a JSONL index")
    config = load_config(
        args.config, args.set, sources_override=[{"path": str(manifest)}]
    )
    if config["model"].get("checkpoint") and args.kind != "wan":
        raise ValueError("Native Qwen caches require native qwen_path weights")
    rows = read_manifest(manifest)
    originals = {str(row["sample_id"]): original_digest(row) for row in rows}
    if len(originals) != len(rows):
        raise ValueError("Duplicate sample IDs")
    if any("videos" in row for row in rows):
        raise ValueError(
            "In-place extraction expects flat task-neutral video_path/frames_path rows"
        )
    state_dir = manifest.parent / "metadata" / "latent_runs"
    baseline_path = state_dir / "original_fields.json"
    with file_lock(state_dir / "baseline.lock"):
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())["samples"]
            if baseline != originals or list(baseline) != list(originals):
                raise ValueError(
                    "Original manifest fields differ from extraction baseline"
                )
        else:
            atomic_write(
                baseline_path,
                json.dumps(
                    {
                        "index_sha256_before_latents": file_sha256(manifest),
                        "samples": originals,
                    },
                    indent=2,
                ),
            )
    selected = (
        set(json.loads(Path(args.sample_ids).read_text())) if args.sample_ids else None
    )
    if selected is not None and selected - set(originals):
        raise ValueError("Requested sample IDs are absent from this index")
    work = [
        (i, row)
        for i, row in enumerate(rows)
        if selected is None or row["sample_id"] in selected
    ]
    if args.limit:
        work = work[: args.limit]
    work = [(i, row) for i, row in work if i % args.num_shards == args.shard_index]
    settings = config["data"]["video"]
    contracts = encoder_contracts(config)
    profile = {
        "format": "umm_latent_profile_v2",
        "video": settings,
        "encoders": contracts,
    }
    profile_id = f"{args.profile_name}-{digest(profile)[:16]}"
    profile_path = (
        manifest.parent / "metadata" / "latent_profiles" / f"{profile_id}.json"
    )
    with file_lock(profile_path.with_suffix(".lock")):
        if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
            raise ValueError("Profile collision")
        if not profile_path.exists():
            atomic_write(
                profile_path, json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
            )
    kinds = ("qwen", "wan") if args.kind == "both" else (args.kind,)
    device = config["training"]["device"]
    register_device(device)
    if device != "cpu":
        getattr(torch, device).set_device(args.device_index)
        device = f"{device}:{args.device_index}"
    processor = load_processor(config)
    encoders = Encoders(config, device)
    root = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else manifest.parent / "latents"
    )
    report_path = state_dir / f"{args.run_id}.shard-{args.shard_index:02d}.json"
    report = dict(
        manifest=str(manifest),
        profile=str(profile_path),
        selected=len(work),
        completed=0,
        encoded={kind: 0 for kind in kinds},
        reused={kind: 0 for kind in kinds},
        verified_online=None,
        complete=False,
        started=time.time(),
        pid=os.getpid(),
    )
    updates, completed_ids = {}, set()
    new_files = 0

    def flush():
        if updates:
            merge_latent_updates(manifest, updates, originals)
            updates.clear()
        report["elapsed_seconds"] = time.time() - report["started"]
        report["index_sha256"] = file_sha256(manifest)
        atomic_write(
            report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )

    try:
        for _, row in work:
            sample_id = str(row["sample_id"])
            video = resolve_video(row, manifest)
            source_path = video.get("frames_path") or video.get("path")
            source_hash = file_sha256(source_path)
            rgb = clip = None
            payloads, info, paths = {}, {}, {}
            for kind in kinds:
                contract = contracts[kind]
                preparation = branch_preprocessing(settings, kind)
                branch = digest({"encoder": contract, "preprocessing": preparation})[
                    :20
                ]
                key = digest({"sample_id": sample_id, "source_sha256": source_hash})
                path = root / kind / branch / key[:2] / f"{key}.pt"
                paths[kind] = path
                # Lock per file as well as the index: overlapping/resumed workers are safe.
                with file_lock(path.with_suffix(".lock")):
                    if path.exists():
                        payload = load_cache(
                            path,
                            kind,
                            contract=contract,
                            preprocessing=settings,
                            sample_id=sample_id,
                        )
                        if payload["metadata"]["clip"]["source_sha256"] != source_hash:
                            raise ValueError("Cache source hash differs")
                        old_info = row.get("latent_info", {}).get(kind)
                        old_path = video.get(f"{kind}_cache")
                        if (
                            old_info
                            and old_path == str(path)
                            and old_info.get("sha256") != file_sha256(path)
                        ):
                            raise ValueError(
                                f"{sample_id}: stored file checksum differs"
                            )
                        report["reused"][kind] += 1
                    else:
                        if args.verify_only:
                            raise FileNotFoundError(f"Missing latent: {path}")
                        if rgb is None:
                            rgb, clip = prepare_video(video, settings)
                            if clip["source_sha256"] != source_hash:
                                raise ValueError("Source changed while decoding")
                        metadata = dict(
                            sample_id=sample_id,
                            clip=clip,
                            preprocessing=preparation,
                            checkpoint=contract["weights"],
                            encoder=contract,
                        )
                        if kind == "qwen":
                            pixels, grid, times = prepare_qwen_inputs(
                                rgb, clip, processor, settings
                            )
                            tensor = encoders.encode(kind, rgb, pixels, grid)
                            metadata["qwen_timestamps"] = times
                            payload = dict(
                                format=FORMATS_V2[kind],
                                features=tensor,
                                grid_thw=grid,
                                metadata=metadata,
                            )
                        else:
                            tensor = encoders.encode(kind, rgb)
                            payload = dict(
                                format=FORMATS_V2[kind],
                                latent=tensor,
                                metadata=metadata,
                            )
                        metadata["tensor"] = tensor_info(tensor)
                        save_cache(path, payload)
                        # Round trip through the exact training loader, preserving dtype and values.
                        restored = load_cache(
                            path,
                            kind,
                            contract=contract,
                            preprocessing=settings,
                            sample_id=sample_id,
                        )
                        key_name = "features" if kind == "qwen" else "latent"
                        torch.testing.assert_close(
                            restored[key_name], tensor, rtol=0, atol=0
                        )
                        if restored[key_name].dtype != tensor.dtype:
                            raise ValueError("Serialization changed tensor dtype")
                        report["encoded"][kind] += 1
                        new_files += 1
                        if args.stop_after_files and new_files >= args.stop_after_files:
                            raise RuntimeError(
                                "Intentional interruption after committed latent file"
                            )
                payloads[kind] = payload
                info[kind] = {
                    **payload["metadata"]["tensor"],
                    "format": payload["format"],
                    "sha256": file_sha256(path),
                    "file_bytes": path.stat().st_size,
                }
                if kind == "qwen":
                    info[kind].update(
                        grid_thw=payload["grid_thw"].tolist(),
                        num_sampled_frames=len(payload["metadata"]["qwen_timestamps"]),
                    )
            if (
                len(payloads) == 2
                and payloads["qwen"]["metadata"]["clip"]
                != payloads["wan"]["metadata"]["clip"]
            ):
                raise ValueError("Qwen/Wan clips differ")
            lengths = (
                _validate_sequences(row, payloads["qwen"], processor, config)
                if "qwen" in payloads
                else {}
            )
            if lengths:
                info["qwen"]["sequence_lengths"] = lengths
            updates[sample_id] = {
                f"{kind}_cache": os.path.relpath(path, manifest.parent)
                for kind, path in paths.items()
            }
            updates[sample_id].update(
                latent_info=info,
                latent_profile=os.path.relpath(profile_path, manifest.parent),
            )
            completed_ids.add(sample_id)
            report["completed"] += 1
            if report["completed"] % args.flush_every == 0:
                flush()
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "completed": report["completed"],
                        "total": len(work),
                        "encoded": report["encoded"],
                        "reused": report["reused"],
                        "elapsed_seconds": round(time.time() - report["started"], 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        flush()
        if args.verify_online:
            if args.kind != "both":
                raise ValueError("Online validation requires both latent kinds")
            report["verified_online"] = _verify_dataset(
                manifest, completed_ids, processor, config, encoders
            )
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        flush()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
