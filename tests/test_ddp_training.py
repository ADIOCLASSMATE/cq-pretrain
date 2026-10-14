"""Two real Gloo ranks validate unused-task parameters across accumulation."""
from copy import deepcopy
from datetime import timedelta
import os
import socket
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from models.modeling_umm import UMM
from training.loss import forward_loss
from training.optimizer import build_optimizer_and_scheduler
from test_model_core import activate_head, config as tiny_config
from test_training import manual_loss, settings, tasks


def _rank_batches(rank, joint=False):
    with torch.random.fork_rng():
        torch.manual_seed(201 + rank)
        batches = tasks()
    for batch in batches:
        text = batch["token_types"].eq(0)
        batch["input_ids"][text] += rank
        labels = batch["labels"]
        labels[labels.ne(-100) & text] += rank
    if not joint:
        return batches
    mixed = deepcopy(batches[1])
    mixed.update(task="joint", labels=batches[3]["labels"].clone())
    single = deepcopy(mixed)
    # The final synchronized microbatch has a different active modality per rank.
    if rank == 0:
        single["image_loss_mask"].zero_()
    else:
        single["labels"].fill_(-100)
    return [mixed, single]


def _reference_loss(model, batch, config, generator):
    if batch["task"] != "joint":
        return manual_loss(model, batch, config, generator)
    selected = batch["labels"].ne(-100)
    hidden = model.backbone(batch)["query_hidden"]
    text = F.cross_entropy(model.lm_head(hidden[selected]), batch["labels"][selected], reduction="sum")
    text = text / selected.sum().clamp_min(1) * config["lambda_text"]
    flow_batch = dict(batch, task="t2i")
    if batch["image_loss_mask"].any():
        flow = manual_loss(model, flow_batch, config, generator)
    else:
        flow_batch["image_loss_mask"] = batch["token_types"].eq(1)
        flow = manual_loss(model, flow_batch, config, generator) * 0
    return text + flow


def _ddp_worker(rank, rendezvous, output, joint):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        # Initialize the same rank-aware RNG owner as the training entrypoint.
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                          LOCAL_WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        torch.manual_seed(14)
        model = UMM(tiny_config())
        activate_head(model.generator)
        reference = deepcopy(model)
        cfg = settings(output)
        optimizer, _ = build_optimizer_and_scheduler(model, cfg)
        ref_optimizer, _ = build_optimizer_and_scheduler(reference, cfg)
        distributed = DistributedDataParallel(model, find_unused_parameters=True)
        set_seed(30, device_specific=True)
        # The explicit reference below still uses seed + rank. Real training
        # must produce those same draws through its global RNG, without a
        # manually seeded generator concealing a wrong entrypoint policy.
        generator = None
        batches = _rank_batches(rank, joint)
        accumulation = len(batches)
        for index, batch in enumerate(batches):
            if index < accumulation - 1:
                with distributed.no_sync():
                    (forward_loss(distributed, batch, cfg, generator)["loss"] / accumulation).backward()
            else:
                (forward_loss(distributed, batch, cfg, generator)["loss"] / accumulation).backward()
        for source_rank in range(2):
            draws = torch.Generator().manual_seed(30 + source_rank)
            for batch in _rank_batches(source_rank, joint):
                (_reference_loss(reference, batch, cfg["model"], draws) / (2 * accumulation)).backward()
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5, msg=name)
        optimizer.step()
        ref_optimizer.step()
        for name, actual in model.state_dict().items():
            torch.testing.assert_close(actual, reference.state_dict()[name], atol=2e-7, rtol=2e-5, msg=name)
        from evaluations.validation import validate_losses
        # Per-source batch partitioning must retain the same seeds and means
        # as one process, even with uneven task lengths.
        probe = [dict(task=task, index=i) for task, count in (("text", 1), ("t2i", 3), ("i2t", 2)) for i in range(count)]
        def validation_forward(model, batch):
            value = torch.rand(())
            return dict(loss=value, raw_loss=value, target_count=batch["index"] + 1)
        before = torch.get_rng_state().clone()
        actual = validate_losses(model, probe, validation_forward,
            schedule=["text", "t2i", "text", "i2t"], distributed=True)
        expected = validate_losses(model, probe, validation_forward,
            schedule=["text", "t2i", "text", "i2t"], distributed=False)
        for key in expected:
            assert actual[key] == pytest.approx(expected[key]), key
        assert torch.equal(torch.get_rng_state(), before)
        accelerator.free_memory()
        if rank == 0:
            torch.save({"ranks": 2, "tasks_per_rank": accumulation}, output / "passed.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize("joint", [False, True])
def test_two_rank_mixed_accumulation_matches_global_reference(tmp_path, joint):
    # Sandboxed CI can prohibit sockets even though PyTorch includes Gloo.
    # Distinguish that environment limit from a numerical/distributed failure.
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local sockets are forbidden; run this test with loopback socket access")
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    rendezvous = (tmp_path / "store").as_uri()
    children = mp.spawn(_ddp_worker, args=(rendezvous, tmp_path, joint), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not children.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Two-rank training exceeded the 60-second deadline")
    finally:
        for process in children.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert torch.load(tmp_path / "passed.pt", weights_only=True) == {"ranks": 2, "tasks_per_rank": 2 if joint else 4}
