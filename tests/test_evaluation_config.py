"""Evaluation tasks and output management are independent of training budgets."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from configs.config import DEFAULTS, load_config
from configs.evaluation import load_tasks, resolve_evaluation_paths
from utils.resume_contract import training_contract, validate_contract
from utils.run_logging import create_attempt, resolve_attempt


def config_at(tmp_path):
    config = deepcopy(DEFAULTS)
    config["training"]["output_dir"] = str(tmp_path / "experiment")
    return config


def write_task(path, **values):
    task = {"name": "preview", "benchmark": "t2v", "actions": ["sample"],
            "data": {}, "sample": {}, "score": {}}
    task.update(values)
    path.write_text(yaml.safe_dump(task))
    return path


def test_eval_config_ignores_training_budget_and_accumulation_only(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"training": {"max_train_steps": 1,
                                                "gradient_accumulation_steps": 1}}))
    config = load_config(path, purpose="eval")
    assert config["training"]["max_train_steps"] == 1
    with pytest.raises(ValueError, match="whole task schedule cycles"):
        load_config(path)
    with pytest.raises(ValueError, match="MindSpeed requires NPU"):
        load_config(path, ["training.device=cpu"], purpose="eval")
    with pytest.raises(ValueError, match="purpose"):
        load_config(path, purpose="unknown")


def test_task_paths_resolve_against_each_yaml_not_working_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    task_dir = run_dir / "tasks"
    task_dir.mkdir(parents=True)
    task_path = write_task(task_dir / "preview.yaml", data={"sources": [{"path": "../data.jsonl"}]})
    run = run_dir / "config.yaml"
    run.write_text(yaml.safe_dump({"evaluation": {"tasks": ["tasks/preview.yaml"]}}))
    monkeypatch.chdir(tmp_path)
    config = load_config(run, purpose="eval")
    assert config["evaluation"]["tasks"] == [str(task_path)]
    task = load_tasks(None, config)[0]
    assert task["_path"] == str(task_path)
    assert task["data"]["sources"][0]["path"] == str(run_dir / "data.jsonl")
    assert config["data"]["sources"] == []


def test_task_schema_names_and_unsupported_trajectories(tmp_path):
    config = config_at(tmp_path)
    path = write_task(tmp_path / "task.yaml")
    assert load_tasks([path], config)[0]["benchmark"] == "t2v"
    with pytest.raises(ValueError, match="Duplicate"):
        load_tasks([path, path], config)
    for updates, match in (
        ({"name": "../escape"}, "single nonempty"),
        ({"actions": ["sample", "sample"]}, "unique"),
        ({"actions": ["loss"]}, "sample/score"),
        ({"typo": 1}, "only"),
        ({"sample": {"collect": {"latent_steps": [1]}}}, "unsupported"),
    ):
        write_task(path, **updates)
        with pytest.raises(ValueError, match=match):
            load_tasks([path], config)


def test_default_task_and_all_presets_load(tmp_path):
    config = config_at(tmp_path)
    assert load_tasks(None, config)[0]["benchmark"] == "loss"
    root = Path(__file__).resolve().parents[1]
    tasks = load_tasks(sorted((root / "configs/evaluations").glob("*.yaml")), config)
    assert {task["benchmark"] for task in tasks} == {"loss", "t2v", "v2t", "vbench"}


def test_attempts_follow_common_log_root_and_exact_eval_overrides(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    paths = resolve_evaluation_paths(config, "step20")
    root = tmp_path / "experiment"
    assert paths["output_dir"] == root / "evaluations/step20"
    assert paths["log_dir"] == root / "logs/eval/step20"
    train = create_attempt(config)
    first = create_attempt(config, purpose="eval", eval_id="step20")
    second = create_attempt(config, purpose="eval", eval_id="step20")
    assert train.parent == root / "logs/train"
    assert first.parent == second.parent == paths["log_dir"] and first != second
    monkeypatch.setenv("UMM_LOG_ATTEMPT", str(first))
    assert resolve_attempt(config, purpose="eval", eval_id="step20") == first
    with pytest.raises(ValueError, match="does not match"):
        resolve_attempt(config)
    config["evaluation"].update(output_dir=str(tmp_path / "results"), log_dir=str(tmp_path / "elog"))
    paths = resolve_evaluation_paths(config, "step20")
    assert paths["output_dir"] == tmp_path / "results"
    assert paths["log_dir"] == tmp_path / "elog"
    config["evaluation"]["output_dir"] = str(root / "checkpoints/exports")
    with pytest.raises(ValueError, match="separate"):
        resolve_evaluation_paths(config, "step20")


def test_eval_management_does_not_change_resume_contract(tmp_path):
    original = config_at(tmp_path)
    changed = deepcopy(original)
    changed["evaluation"].update(tasks=["new.yaml"], eval_id="new", output_dir="results",
                                 log_dir="elog", num_processes=2, max_batches=3)
    assert validate_contract(original, changed, training_contract(original))
    changed["evaluation"]["unknown_future_setting"] = True
    with pytest.raises(ValueError, match="unknown_future_setting"):
        validate_contract(original, changed)


def test_processor_lookup_prefers_explicit_then_adjacent_then_native(tmp_path, monkeypatch):
    from training.train import load_processor
    from transformers import AutoProcessor

    class Tokenizer:
        init_kwargs = {}
        _special_tokens_map = {}
        _extra_special_tokens = []

        def get_vocab(self):
            return {"<|text_mask|>": 59, "<|vision_mask|>": 60}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab()[token]

    class Processor:
        tokenizer = Tokenizer()

    used = []

    def read(path, **kwargs):
        used.append(str(path))
        assert kwargs == {"local_files_only": True}
        return Processor()

    monkeypatch.setattr(AutoProcessor, "from_pretrained", read)
    config = config_at(tmp_path)
    export = tmp_path / "exports"
    (export / "processor").mkdir(parents=True)
    config["model"]["checkpoint"] = str(export / "weights.pt")
    load_processor(config)
    assert used[-1] == str(export / "processor")
    config["model"]["processor_path"] = str(tmp_path / "explicit")
    load_processor(config)
    assert used[-1] == str(tmp_path / "explicit")
    config["model"].pop("processor_path")
    (export / "processor").rmdir()
    load_processor(config)
    assert used[-1] == config["model"]["qwen_path"]


@pytest.mark.parametrize("purpose", ["train", "eval"])
@pytest.mark.parametrize("cap,expected", [(None, 2), (3, 3)])
def test_validation_batch_cap_inherits_legacy_training_setting(tmp_path, purpose, cap, expected):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({
        "training": {"validation_max_batches": 2},
        "evaluation": {"max_batches": cap},
    }))
    config = load_config(path, purpose=purpose)
    assert config["training"]["validation_max_batches"] == 2
    assert config["evaluation"]["max_batches"] == expected


@pytest.mark.parametrize("name", ["run.json", "config.yaml", ".writer.lock"])
def test_task_name_cannot_replace_evaluation_metadata(tmp_path, name):
    path = write_task(tmp_path / "task.yaml", name=name)
    with pytest.raises(ValueError, match="reserved for evaluation metadata"):
        load_tasks([path], config_at(tmp_path))
