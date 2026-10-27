"""Evaluation-task YAML loading and output paths, independent of model imports."""

from copy import deepcopy
from pathlib import Path

import yaml

from configs.config import resolve_output_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_FIELDS = {"name", "benchmark", "actions", "data", "sample", "score"}
RESERVED_TASK_NAMES = {"run.json", "config.yaml", ".writer.lock"}


def directory_id(value, label):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a single nonempty directory name")
    return value


def absolute_path(value, base=PROJECT_ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_evaluation_paths(config, eval_id):
    """Explicit eval paths are exact roots; only the log attempt is appended."""
    directory_id(eval_id, "evaluation.eval_id")
    train = resolve_output_paths(config)
    settings = config.get("evaluation", {})
    output = (
        absolute_path(settings["output_dir"])
        if settings.get("output_dir")
        else train["output_dir"] / "evaluations" / eval_id
    )
    logs = (
        absolute_path(settings["log_dir"])
        if settings.get("log_dir")
        else train["log_dir"] / "eval" / eval_id
    )
    checkpoints = train["checkpoint_dir"]
    common_logs = train["log_dir"]
    if output == common_logs or common_logs in output.parents or output in common_logs.parents:
        raise ValueError("Evaluation results and experiment logs must be separate directory trees")
    for left, right in ((output, logs), (output, checkpoints), (logs, checkpoints)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("Evaluation results, logs and checkpoints must be separate directory trees")
    # A result directory cannot stand in for experiment metadata itself.
    if output == train["output_dir"] or output in train["output_dir"].parents:
        raise ValueError("Evaluation output_dir must not contain the experiment root")
    return {
        "output_dir": output,
        "log_dir": logs,
        "experiment_dir": train["output_dir"],
        "checkpoint_dir": checkpoints,
    }


def _task_paths(task, base):
    """Resolve only file-valued fields; task IDs and Python commands are not paths."""
    data = task["data"]
    for key in ("path", "manifest", "prompts_path"):
        if data.get(key):
            data[key] = str(absolute_path(data[key], base))
    sources = data.get("sources")
    if sources is not None:
        if not isinstance(sources, list):
            raise ValueError("task data.sources must be a list")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError("Each task source requires a path")
            source["path"] = str(absolute_path(source["path"], base))
    score = task["score"]
    for key in ("vbench_root", "run_dir", "manifest"):
        if score.get(key):
            score[key] = str(absolute_path(score[key], base))
    # Bare `python` is a PATH lookup, whereas a relative executable is YAML-local.
    if score.get("python") and ("/" in score["python"] or score["python"].startswith("~")):
        score["python"] = str(absolute_path(score["python"], base))


def validate_task(task):
    """Validate common syntax; adapters validate benchmark-specific fields."""
    if not isinstance(task, dict) or set(task) - TASK_FIELDS:
        raise ValueError(f"Task YAML must contain only {sorted(TASK_FIELDS)}")
    directory_id(task.get("name"), "task name")
    if task["name"] in RESERVED_TASK_NAMES:
        raise ValueError(f"Task name is reserved for evaluation metadata: {task['name']}")
    directory_id(task.get("benchmark"), "task benchmark")
    actions = task.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or any(action not in {"sample", "score"} for action in actions)
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("Task actions must be a nonempty unique list of sample/score")
    for key in ("data", "sample", "score"):
        if task.get(key) is None:
            task[key] = {}
        if not isinstance(task[key], dict):
            raise ValueError(f"Task {key} must be a mapping or null")
    collect = task["sample"].get("collect", {})
    if isinstance(collect, dict) and any(
        collect.get(key) for key in ("latent_steps", "trajectory", "step_interval")
    ):
        raise ValueError("Intermediate video trajectories are unsupported; models are unchanged")
    return task


def load_tasks(paths, config):
    """Load task files; CLI paths use cwd, config tasks were resolved by load_config.

    References inside a task YAML use that file's parent directory. Inherited
    data.validation_sources preserve the existing project-relative data contract.
    """
    paths = paths or config.get("evaluation", {}).get("tasks", [])
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Evaluation task paths must be a list")
    paths = list(paths)
    if not paths:
        paths = [PROJECT_ROOT / "configs/evaluations/loss.yaml"]
    tasks = []
    seen = set()
    for filename in paths:
        path = Path(filename).expanduser().resolve()
        task = validate_task(deepcopy(yaml.safe_load(path.read_text())))
        if task["name"] in seen:
            raise ValueError(f"Duplicate evaluation task name: {task['name']}")
        seen.add(task["name"])
        _task_paths(task, path.parent)
        task["_path"] = str(path)
        tasks.append(task)
    return tasks
