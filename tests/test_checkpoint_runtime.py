from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.ema import RankShardedEMA, build_sharded_ema_layout, merge_sharded_ema_state_dict
from utils.checkpoint import save_checkpoint, load_checkpoint, export_weights, validate_checkpoint


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 3)
        self.output = nn.Linear(3, 7, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("counter", torch.tensor(0))

    def forward(self, indices):
        return self.output(self.embedding(indices)).square().mean()


def make_training():
    torch.manual_seed(4)
    model = TiedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    ema = RankShardedEMA(build_sharded_ema_layout(model, world_size=1, chunk_numel=5),
                         rank=0, decay=0.9, update_after_step=0)
    ema.bind(model)
    ema.initialize_from_model()
    return model, optimizer, scheduler, ema


def train_step(model, optimizer, scheduler, ema, step):
    optimizer.zero_grad()
    loss = model(torch.randint(0, 7, (5,)))
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.counter.add_(1)
    ema.maybe_update(step)
    return loss.detach()


def test_complete_resume_matches_uninterrupted_optimizer_step(tmp_path):
    model, optimizer, scheduler, ema = make_training()
    train_step(model, optimizer, scheduler, ema, 1)
    path = tmp_path / "checkpoint-1"
    config = {"schedule": ["text", "t2i"], "lr": 0.01}
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                    global_step=1, data_state={"consumed": 11}, task_state={"position": 0}, ema=ema, config=config)
    expected_python = random.random()
    expected = train_step(model, optimizer, scheduler, ema, 2)
    expected_state = {k: v.clone() for k, v in model.state_dict().items()}
    expected_shards = {k: v.clone() for k, v in ema.shards.items()}
    restored, opt2, sched2, ema2 = make_training()
    resume = load_checkpoint(path, model=restored, optimizer=opt2, scheduler=sched2,
                             ema=ema2, expected_config=config)
    assert resume["global_step"] == 1 and resume["data_state"] == {"consumed": 11}
    assert resume["task_state"] == {"position": 0}
    assert random.random() == expected_python
    actual = train_step(restored, opt2, sched2, ema2, 2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert sched2.state_dict() == scheduler.state_dict()
    assert restored.output.weight is restored.embedding.weight
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
    for name, value in ema2.shards.items():
        torch.testing.assert_close(value, expected_shards[name], atol=0, rtol=0)
    raw = export_weights(tmp_path / "raw.pt", model=restored, config=config)
    exported = torch.load(raw, weights_only=True)
    assert exported["weights"] == "raw"
    ema_path = export_weights(tmp_path / "ema.pt", ema_checkpoint=path / "ema", config=config)
    assert torch.load(ema_path, weights_only=True)["weights"] == "ema"




def test_legacy_single_draw_checkpoint_accepts_default_repeats(tmp_path):
    model, optimizer, scheduler, _ = make_training()
    path = tmp_path / "legacy-video"
    old_config = {"flow": {"time_sampling": "logit_normal"}}
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                    global_step=0, data_state={}, config=old_config)
    config = {"flow": {"time_sampling": "logit_normal", "repeats": 1}}
    validate_checkpoint(path, expected_config=config)
    assert "repeats" not in old_config["flow"]
    config["flow"]["repeats"] = 2
    with pytest.raises(ValueError, match="configuration contract differs"):
        validate_checkpoint(path, expected_config=config)


def test_transaction_rejects_incomplete_overwrite_and_different_contract(tmp_path):
    model, optimizer, scheduler, ema = make_training()
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, global_step=0,
                    data_state={}, ema=ema, config={"x": 1})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, global_step=0,
                        data_state={}, ema=ema)
    with pytest.raises(ValueError, match="world size"):
        validate_checkpoint(path, world_size=2)
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
                        expected_config={"x": 2})
    (path / "training_state.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="truncated"):
        validate_checkpoint(path)


def test_accelerator_checkpoint_restores_registered_optimizer_and_scheduler(tmp_path):
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, optimizer, scheduler, _ = make_training()
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def step():
        optimizer.zero_grad()
        loss = model(torch.randint(0, 7, (5,)))
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        return loss.detach()

    step()
    path = tmp_path / "accelerated"
    save_checkpoint(path, model=accelerator.unwrap_model(model), optimizer=optimizer,
                    scheduler=scheduler, global_step=1, data_state={"cursor": 1}, accelerator=accelerator)
    expected_loss = step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    load_checkpoint(path, model=accelerator.unwrap_model(model), optimizer=optimizer,
                    scheduler=scheduler, accelerator=accelerator)
    torch.testing.assert_close(step(), expected_loss, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


def _distributed_ema_worker(rank, init_path, checkpoint):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        model = TiedModel()
        original = model.embedding.weight.detach().clone()
        layout = build_sharded_ema_layout(model, world_size=2, chunk_numel=4)
        ema = RankShardedEMA(layout, rank=rank, decay=0.5, update_after_step=2)
        ema.bind(model)
        ema.initialize_from_model()
        with torch.no_grad():
            model.embedding.weight.add_(2)
        assert not ema.maybe_update(1)
        assert ema.maybe_update(2)  # delayed start copies current model
        with torch.no_grad():
            model.embedding.weight.add_(4)
        ema.maybe_update(3)
        live = model.embedding.weight.detach().clone()
        try:
            with ema.applied_to(model):
                torch.testing.assert_close(model.embedding.weight, original + 4)
                assert model.embedding.weight is model.output.weight
                raise ValueError("restore after evaluation exception")
        except ValueError:
            pass
        torch.testing.assert_close(model.embedding.weight, live)
        runtime = SimpleNamespace(num_processes=2, process_index=rank, is_main_process=rank == 0)
        ema.save_checkpoint(checkpoint, runtime, global_step=3)
        replacement = RankShardedEMA(layout, rank=rank, decay=0.5, update_after_step=2)
        replacement.bind(model)
        replacement.load_checkpoint(checkpoint, runtime, expected_global_step=3)
        for name in ema.shards:
            torch.testing.assert_close(ema.shards[name], replacement.shards[name])
        if rank == 0:
            merged = merge_sharded_ema_state_dict(checkpoint)
            torch.testing.assert_close(merged["embedding.weight"], original + 4)
            assert merged["embedding.weight"] is merged["output.weight"]
    finally:
        dist.destroy_process_group()


def test_rank_sharded_ema_real_two_process_collectives(tmp_path):
    torch.multiprocessing.spawn(_distributed_ema_worker,
        args=(str(tmp_path / "dist-init"), str(tmp_path / "ema")), nprocs=2, join=True)
