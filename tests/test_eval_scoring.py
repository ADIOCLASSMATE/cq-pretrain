"""Unified scorer contracts, exact validation parity and inference data views."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from data.dataloader import build_dataloaders, build_validation_loader
from data.dataset import VideoDataset, read_records
from evaluations.scoring import get_scorer, score_loss
from evaluations.validation import validate_losses, loss_validation_options
from training.loss import prepare_batch, forward_loss
from tests.video_fixtures import make_fixture


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    return make_fixture(tmp_path / "fixture")


def test_unlabeled_v2t_prompt_matches_supervised_record_without_answer(fixture, tmp_path):
    config, processor, _, rows, _ = fixture
    labeled = rows[2]
    unlabeled = deepcopy(labeled)
    unlabeled.pop("caption")
    path = tmp_path / "unlabeled.jsonl"
    path.write_text(json.dumps(unlabeled) + "\n")
    source = {"path": str(path)}
    with pytest.raises(ValueError, match="nonempty caption"):
        read_records(source)
    records = read_records(source, prompt_only=True)
    assert records[0]["messages"][-1]["role"] == "user"
    actual = VideoDataset(source, processor, config, prompt_only=True)[0]
    expected = VideoDataset(config["data"]["sources"][0], processor, config, prompt_only=True)[2]
    assert torch.equal(actual["input_ids"], expected["input_ids"])
    assert bool(actual["labels"].eq(-100).all())
    assert "wan_latent" not in actual


def test_unlabeled_messages_do_not_require_fake_assistant(fixture, tmp_path):
    config, processor, _, _, _ = fixture
    row = read_records(config["data"]["sources"][0])[2]
    row["messages"] = row["messages"][:-1]
    path = tmp_path / "messages.jsonl"
    path.write_text(json.dumps(row) + "\n")
    source = {"path": str(path)}
    with pytest.raises(ValueError, match="assistant target"):
        read_records(source)
    sample = VideoDataset(source, processor, config, prompt_only=True)[0]
    assert bool(sample["labels"].eq(-100).all())


def test_validation_loader_needs_no_training_data_and_preserves_batches(fixture):
    config, processor, _, _, _ = fixture
    config["data"]["validation_sources"] = deepcopy(config["data"]["sources"])
    _, training_validation = build_dataloaders(config, processor)
    original = list(training_validation)
    config["data"]["sources"] = []
    standalone = list(build_validation_loader(config, processor))
    assert len(original) == len(standalone) == 4
    for old, new in zip(original, standalone):
        assert old["task"] == new["task"]
        assert old["sample_ids"] == new["sample_ids"]
        for name in ("input_ids", "labels", "mm_token_type_ids", "target_video_ids"):
            assert torch.equal(old[name], new[name])


def test_loss_scorer_matches_training_validation_and_preserves_state(fixture, tmp_path):
    config, processor, model, _, _ = fixture
    config["data"]["validation_sources"] = deepcopy(config["data"]["sources"])
    config["model"]["video_uncond_prob"] = 0.5
    _, loader = build_dataloaders(config, processor)
    observed = []

    def evaluate(current, batch):
        output = forward_loss(current, prepare_batch(batch, "cpu", config), config)
        observed.append({key: float(value) for key, value in output.items()})
        return output

    expected = validate_losses(model, loader, evaluate, schedule=config["data"]["schedule"],
                               **loss_validation_options(config))
    runtime = SimpleNamespace(config=config, processor=processor, device=torch.device("cpu"),
                              get_model=lambda: model, autocast=nullcontext)
    rng = torch.get_rng_state().clone()
    modes = [item.training for item in model.modules()]
    actual = score_loss(runtime, {"score": {"save_batches": True, "seed": None, "max_batches": None}},
                        tmp_path / "score")
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [item.training for item in model.modules()]
    rows = [json.loads(line) for line in (tmp_path / "score/batches.rank-0.jsonl").read_text().splitlines()]
    assert [(row["task"], row["batch_index"]) for row in rows] == [
        ("t2v", 0), ("t2v", 1), ("v2t", 0), ("v2t", 1)]
    for row, old in zip(rows, observed):
        assert {key: row[key] for key in old} == old
    assert json.loads((tmp_path / "score/metrics.json").read_text()) == expected


def test_scorer_rejects_unknown_or_unimplemented_protocols(fixture, tmp_path):
    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("not_registered")
    config, processor, model, _, _ = fixture
    runtime = SimpleNamespace(config=config, processor=processor, device=torch.device("cpu"),
                              get_model=lambda: model, autocast=nullcontext)
    with pytest.raises(ValueError, match="not offline statistics"):
        score_loss(runtime, {"score": {"input": "collected"}}, tmp_path)
    with pytest.raises(ValueError, match="objective must be training"):
        score_loss(runtime, {"score": {"objective": "sampled"}}, tmp_path)


def test_vbench_planner_preserves_native_protocol_and_external_score_interface(fixture, tmp_path):
    from evaluations.benchmarks.vbench.adapter import plan_vbench, prepare_vbench, vbench_score_command
    from evaluations.benchmarks.vbench.generate import load_prompts, make_tasks
    from evaluations.benchmarks.vbench.score import read_run

    config, _, _, _, _ = fixture
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"identity-only")
    config["model"]["checkpoint"] = str(checkpoint)
    task = {"data": {"split": "subset"}, "sample": {"seed": 73},
            "score": {"python": "/opt/cuda/bin/python", "vbench_root": "/opt/VBench"}}
    requests, metadata, prompts = plan_vbench(task, config)
    native = make_tasks(load_prompts("subset")[0], 73)
    assert len(requests) == len({item["id"] for item in requests}) == 150
    for request, expected in zip(requests, native):
        assert all(request[key] == value for key, value in expected.items())
    directory = tmp_path / "vbench"
    prepare_vbench(directory, metadata, prompts)
    run, _ = read_run(directory)
    assert run["config"]["sampling"]["seed"] == 73
    command = vbench_score_command(task, directory)
    assert command[:3] == ["/opt/cuda/bin/python", "-m", "evaluations.benchmarks.vbench.score"]
    assert Path(command[command.index("--run-dir") + 1]) == directory
    with pytest.raises(ValueError, match="exactly 5"):
        plan_vbench({"sample": {"samples_per_prompt": 1}}, config)


def test_external_score_logs_do_not_pollute_dispatcher_json(tmp_path, monkeypatch, capsys):
    from evaluations.benchmarks.vbench import adapter

    command = [sys.executable, "-c", "import sys; print('official stdout'); print('official stderr', file=sys.stderr)"]
    monkeypatch.setattr(adapter, "vbench_score_command", lambda *_: command)
    log_dir = tmp_path / "attempt/vbench"
    result = adapter.run_vbench_score({"score": {"dry_run": True}, "_score_log_dir": str(log_dir)},
                                      tmp_path / "results")
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "official stdout" in streams.err and "official stderr" in streams.err
    assert result["status"] == "dry_run"
    log = (log_dir / "console.log").read_text()
    assert "official stdout" in log and "official stderr" in log


def test_vbench_requires_video_protocol_and_inherits_null_seed(fixture, tmp_path):
    from evaluations.benchmarks.vbench.adapter import plan_vbench

    config = fixture[0]
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'identity-only')
    config['model']['checkpoint'] = str(checkpoint)
    requests, _, _ = plan_vbench({'sample': {'seed': None}}, config)
    assert requests[0]['seed'] == config['sampling']['seed']
    with pytest.raises(ValueError, match='requires MP4'):
        plan_vbench({'sample': {'outputs': ['latent']}}, config)
    with pytest.raises(ValueError, match='Unsupported VBench sample'):
        plan_vbench({'sample': {'max_samples': 1}}, config)
