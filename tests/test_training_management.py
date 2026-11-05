"""Recovery/retention failures must leave previous checkpoints and logs usable."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from configs.config import DEFAULTS, resolve_output_paths, validate_config
from utils.checkpoint import save_checkpoint, validate_checkpoint
from utils.checkpoint_catalog import (
    DirectoryLock,
    commit_and_prune,
    discover,
    latest,
    quarantine_invalid,
)
from utils.resume_contract import training_contract, validate_contract


def checkpoint(root, step, config=None):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    return save_checkpoint(
        root / f"checkpoint-{step}",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        global_step=step,
        data_state={},
        config=config,
    )


def test_paths_are_experiment_scoped_and_independently_overridable(
    tmp_path, monkeypatch
):
    config = deepcopy(DEFAULTS)
    config["training"]["exp_id"] = "test-exp"
    monkeypatch.chdir(tmp_path)
    project = Path(__file__).resolve().parents[1]
    paths = resolve_output_paths(config)
    assert paths["output_dir"] == project / "outputs/test-exp"
    assert paths["log_dir"] == paths["output_dir"] / "logs"
    assert paths["checkpoint_dir"] == paths["output_dir"] / "checkpoints"
    config["training"].update(
        output_dir=str(tmp_path / "full-dir"), log_dir=str(tmp_path / "logs")
    )
    paths = resolve_output_paths(config)
    assert paths["output_dir"] == tmp_path / "full-dir"
    assert paths["log_dir"] == tmp_path / "logs"
    assert paths["checkpoint_dir"] == tmp_path / "full-dir/checkpoints"
    config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
    assert resolve_output_paths(config)["checkpoint_dir"] == tmp_path / "ckpts"
    config["training"]["log_dir"] = str(tmp_path / "ckpts/logs")
    with pytest.raises(ValueError, match="separate"):
        resolve_output_paths(config)


def test_management_can_change_but_training_and_unknown_fields_are_strict():
    saved = deepcopy(DEFAULTS)
    current = deepcopy(saved)
    current["training"].update(
        exp_id="new",
        output_root="/new-root",
        output_dir="/new",
        log_dir="/logs",
        checkpoint_dir="/weights",
        keep_checkpoints=1,
        save_every=2,
        log_every=20,
        val_every=100,
        validation_max_batches=2,
    )
    current["data"]["validation_sources"] = [{"path": "new-val.jsonl"}]
    current["sampling"]["steps"] = 4
    current["evaluation"]["seed"] = 1
    changes = validate_contract(saved, current, training_contract(saved))
    assert {c["field"] for c in changes} >= {"training.exp_id", "training.log_dir"}
    for section, key, value in (
        ("training", "max_train_steps", 20),
        ("optimizer", "backbone_lr", 0.1),
        ("flow", "repeats", 2),
        ("training", "dataloader_workers", 7),
        ("sampling", "unknown_future_option", True),
        ("model", "freeze_vision", False),
        ("model", "qwen_gdn_backend", "mindspeed"),
    ):
        changed = deepcopy(current)
        changed[section][key] = value
        with pytest.raises(ValueError, match=rf"{section}\.{key}"):
            validate_contract(saved, changed)


def test_missing_gdn_configuration_and_old_contracts_are_rejected():
    saved = deepcopy(DEFAULTS)
    saved["model"].pop("qwen_gdn_backend")
    current = deepcopy(DEFAULTS)
    old_contract = training_contract(saved)
    with pytest.raises(ValueError, match="qwen_gdn_backend"):
        validate_contract(saved, current, old_contract)
    tampered = deepcopy(old_contract)
    tampered["schema"] = "umm_resume_contract_v1"
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_contract(saved, current, tampered)


@pytest.mark.parametrize("backend,device,precision,valid", [
    ("ascendc", "cpu", "no", True),
    ("mindspeed", "npu", "bf16", True),
    ("mindspeed", "cpu", "bf16", False),
    ("mindspeed", "npu", "no", False),
    ("auto", "npu", "bf16", False),
])
def test_gdn_config_requires_explicit_supported_backend(backend, device, precision, valid):
    config = deepcopy(DEFAULTS)
    # Isolate GDN validation from the production distributed-runtime contract.
    config["distributed"]["backend"] = "accelerate"
    config["model"]["qwen_gdn_backend"] = backend
    config["training"].update(device=device, mixed_precision=precision)
    if valid:
        validate_config(config)
    else:
        with pytest.raises(ValueError, match="GDN|qwen_gdn_backend"):
            validate_config(config)


def test_cpu_regression_runtime_rejects_accelerator_training_and_deepspeed(monkeypatch):
    from training.train import build_accelerator

    config = deepcopy(DEFAULTS)
    with pytest.raises(ValueError, match="only for CPU regression"):
        build_accelerator(config)
    config["training"].update(device="cpu", mixed_precision="no")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    with pytest.raises(ValueError, match="DeepSpeed training is no longer supported"):
        build_accelerator(config)


def test_catalog_numeric_order_fallback_quarantine_and_prune(tmp_path):
    root = tmp_path / "checkpoints"
    with DirectoryLock(root):
        first = checkpoint(root, 2)
        commit_and_prune(first, 2)
        second = checkpoint(root, 10)
        commit_and_prune(second, 2)
        third = checkpoint(root, 100)
        (third / "training_state.pt").write_bytes(b"truncated")
        (root / "latest.json").write_text('{"path":"checkpoint-100"}')
        selected, rejected = latest(root)
        assert Path(selected) == second and len(rejected) == 1
        assert first.exists() and third.exists()  # discovery never deletes
        moved = quarantine_invalid(root)
        assert Path(moved[0]["to"]).exists() and not third.exists()
        # Re-saving the corrupt step after fallback must succeed.
        replacement = checkpoint(root, 100)
        exports = root / "exports"
        exports.mkdir()
        (exports / "weights.pt").write_bytes(b"export")
        (root / "unrelated").mkdir()
        removed = commit_and_prune(replacement, 2)
        assert removed == [str(first)] and second.exists() and replacement.exists()
        assert (exports / "weights.pt").read_bytes() == b"export"
        assert (root / "unrelated").exists() and Path(moved[0]["to"]).exists()
        assert json.loads((root / "latest.json").read_text())["global_step"] == 100
        assert commit_and_prune(replacement, None) == []


def test_no_silent_restart_after_committed_history_and_no_symlink_deletion(tmp_path):
    root = tmp_path / "checkpoints"
    assert latest(root) == (None, [])
    good = checkpoint(root, 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "checkpoint-2").symlink_to(outside, target_is_directory=True)
    assert Path(latest(root)[0]) == good
    commit_and_prune(good, 1)
    assert outside.is_dir() and (root / "checkpoint-2").is_symlink()
    (good / "training_state.pt").unlink()
    with pytest.raises(RuntimeError, match="refusing"):
        latest(root)


@pytest.mark.parametrize(
    "filename", ["metadata.json", "checkpoint_complete.json", "checkpoint_files.json"]
)
def test_malformed_json_metadata_falls_back_instead_of_crashing_discovery(
    tmp_path, filename
):
    first = checkpoint(tmp_path, 1)
    second = checkpoint(tmp_path, 2)
    (second / filename).write_text("[]")
    path, rejected = latest(tmp_path)
    assert Path(path) == first
    assert len(rejected) == 1 and "JSON objects" in rejected[0]["reason"]


def test_failed_save_keeps_previous_and_writer_lock_is_exclusive(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    with DirectoryLock(root):
        old = checkpoint(root, 1)
        commit_and_prune(old, 1)
        with pytest.raises(RuntimeError, match="already owns"):
            with DirectoryLock(root):
                pass

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="disk full"):
            checkpoint(root, 2)
        assert Path(latest(root)[0]) == old
        validate_checkpoint(old)
        assert not (root / "checkpoint-2").exists()
    with DirectoryLock(root):
        pass


def test_legacy_metadata_and_runtime_topology_validation(tmp_path):
    config = deepcopy(DEFAULTS)
    path = checkpoint(tmp_path, 1, config)
    changed = deepcopy(config)
    changed["training"].update(log_dir="/new/logs", keep_checkpoints=1)
    validate_checkpoint(path, expected_config=changed)
    with pytest.raises(ValueError, match="topology"):
        validate_checkpoint(
            path, expected_runtime={"backend": "torch", "world_size": 4}
        )
    metadata = json.loads((path / "metadata.json").read_text())
    metadata.pop("resume_contract")
    metadata.pop("runtime_contract")
    (path / "metadata.json").write_text(json.dumps(metadata))
    inventory = json.loads((path / "checkpoint_files.json").read_text())
    inventory["metadata.json"] = (path / "metadata.json").stat().st_size
    (path / "checkpoint_files.json").write_text(json.dumps(inventory))
    validate_checkpoint(path, expected_config=changed)
    changed["optimizer"]["eps"] = 0.01
    with pytest.raises(ValueError, match="optimizer.eps"):
        validate_checkpoint(path, expected_config=changed)


@pytest.mark.parametrize("keep", [0, -1, True, 1.5])
def test_invalid_retention_rejected(keep):
    config = deepcopy(DEFAULTS)
    config["training"]["keep_checkpoints"] = keep
    with pytest.raises(ValueError, match="keep_checkpoints"):
        validate_config(config)
