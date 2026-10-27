"""CLI scheduler, artifact integrity and four-process resume contracts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from evaluations.artifacts import commit_receipt, read_receipt
from evaluations.eval import main
from tests.video_fixtures import make_fixture
from utils.checkpoint import export_weights


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    config, processor, model, _, _ = make_fixture(tmp_path / "fixture")
    config["data"]["validation_sources"] = deepcopy(config["data"]["sources"])
    config["sampling"].update(steps=1, cfg=1.0, max_new_tokens=2)
    config["evaluation"]["num_processes"] = 4
    config["model"]["checkpoint"] = str(tmp_path / "weights.pt")
    export_weights(tmp_path / "weights.pt", model=model, config=config)
    model_config = tmp_path / "config.yaml"
    model_config.write_text(yaml.safe_dump(config))
    task = dict(name="preview", benchmark="t2v", actions=["sample"],
                data={"prompts": ["blue block", "red block", "blue block moves", "red block moves"]},
                sample={"outputs": ["latent"], "seed": 123}, score={})
    task_config = tmp_path / "preview.yaml"
    task_config.write_text(yaml.safe_dump(task))
    return config, model_config, task_config


def invoke(config, task, *extra, success=True):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    command = [sys.executable, "-m", "evaluations.eval", "--config", str(config),
               "--task-config", str(task), *map(str, extra)]
    result = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=240)
    if success:
        assert result.returncode == 0, result.stderr[-8000:]
        return json.loads(result.stdout)
    assert result.returncode, result.stdout
    return result


def manifest(root):
    return [json.loads(line) for line in (root / "preview/manifest.jsonl").read_text().splitlines()]


def test_four_process_sampling_resume_corruption_repair_and_config_guard(prepared):
    config, path, task = prepared
    base = Path(config["training"]["output_dir"]) / "evaluations"
    continuous = invoke(path, task, "--eval-id", "continuous")
    assert continuous["status"] == "completed"
    original = manifest(base / "continuous")
    assert len(original) == len({row["id"] for row in original}) == 4
    assert {row["rank"] for row in original} == {0, 1, 2, 3}
    partial = invoke(path, task, "--eval-id", "resumed", "--max-new-samples", "2")
    assert partial["status"] == "incomplete"
    partial_rows = manifest(base / "resumed")
    assert len(partial_rows) == 2
    existing = {row["id"]: (base / "resumed/preview" / row["artifacts"]["latent"]).stat().st_mtime_ns
                for row in partial_rows}
    resumed = invoke(path, task, "--run-dir", base / "resumed", "--resume")
    assert resumed["status"] == "completed"
    rows = manifest(base / "resumed")
    assert len(rows) == 4
    for before, after in zip(original, rows):
        assert before["id"] == after["id"]
        a = torch.load(base / "continuous/preview" / before["artifacts"]["latent"], weights_only=True)["latent"]
        b = torch.load(base / "resumed/preview" / after["artifacts"]["latent"], weights_only=True)["latent"]
        assert torch.equal(a, b)
        if after["id"] in existing:
            assert (base / "resumed/preview" / after["artifacts"]["latent"]).stat().st_mtime_ns == existing[after["id"]]
    damaged = base / "resumed/preview" / rows[0]["artifacts"]["latent"]
    expected = torch.load(damaged, weights_only=True)["latent"]
    damaged.write_bytes(b"interrupted")
    repaired = invoke(path, task, "--run-dir", base / "resumed", "--resume",
                      "--set", "evaluation.num_processes=1")
    assert repaired["status"] == "completed"
    assert torch.equal(expected, torch.load(damaged, weights_only=True)["latent"])
    run = json.loads((base / "resumed/run.json").read_text())
    assert len(run["attempts"]) == 3
    assert len(set(run["attempts"])) == 3
    assert all((Path(attempt) / "console.log").exists() for attempt in run["attempts"])
    changed = invoke(path, task, "--run-dir", base / "resumed", "--resume",
                     "--set", "sampling.steps=2", success=False)
    assert "changed" in changed.stderr
    # Same filename and same byte count cannot hide replaced weights.
    checkpoint = Path(config["model"]["checkpoint"])
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    changed = invoke(path, task, "--run-dir", base / "resumed", "--resume", success=False)
    assert "changed" in changed.stderr


def test_dry_run_does_not_create_experiment_and_score_environment_is_lazy(prepared, capsys):
    config, path, task = prepared
    base = Path(config["training"]["output_dir"])
    main(["--config", str(path), "--task-config", str(task), "--dry-run", "--eval-id", "dry"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["needs_model"] is True
    assert not (base / "evaluations/dry").exists()
    score_task = task.with_name("score.yaml")
    score_task.write_text(yaml.safe_dump(dict(name="vbench", benchmark="vbench", actions=["score"],
                                             data={"split": "subset"}, sample={}, score={})))
    main(["--config", str(path), "--task-config", str(score_task), "--print-environment"])
    assert capsys.readouterr().out.strip() == "score"


def test_receipt_rejects_corruption_request_drift_and_directory_escape(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "sample.txt").write_text("ok")
    request = {"id": "one", "seed": 1}
    result = {"id": "one", "artifacts": {"text": "sample.txt"}}
    commit_receipt(task, request, result, rank=0, attempt_id="one")
    assert read_receipt(task, request) is not None
    assert read_receipt(task, dict(request, seed=2)) is None
    (task / "sample.txt").write_text("changed")
    assert read_receipt(task, request) is None
    with pytest.raises(ValueError, match="escapes"):
        commit_receipt(task, request, {"artifacts": {"text": "../outside"}}, rank=0, attempt_id="one")


def test_failed_worker_receipts_resume_without_replaying_completed_requests(prepared, monkeypatch, capsys):
    import evaluations.eval as dispatcher

    config, path, task = prepared
    calls = []

    def execute(plan, count, attempt):
        for job in plan["jobs"]:
            for request in job["pending"]:
                calls.append(request["id"])
                output = Path(job["directory"])
                name = request["id"] + ".pt"
                (output / name).write_text(request["id"])
                commit_receipt(output, request, {"id": request["id"], "artifacts": {"latent": name}},
                               rank=0, attempt_id=attempt.name)
                if len(calls) == 1:
                    raise RuntimeError("simulated interrupted worker after commit")

    monkeypatch.setattr(dispatcher, "launch", execute)
    args = ["--config", str(path), "--task-config", str(task), "--eval-id", "failed"]
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        main(args)
    root = Path(config["training"]["output_dir"]) / "evaluations/failed"
    assert json.loads((root / "run.json").read_text())["status"] == "failed"
    capsys.readouterr()
    main(args + ["--resume"])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert len(calls) == len(set(calls)) == 4


def test_external_score_only_works_after_weights_and_source_manifests_are_removed(prepared, monkeypatch, capsys):
    import evaluations.eval as dispatcher
    import evaluations.scoring as scoring

    config, path, task_path = prepared
    task_path.write_text(yaml.safe_dump(dict(name="vbench", benchmark="vbench", actions=["sample"],
                                           data={"split": "subset"}, sample={}, score={})))

    def prepare(config, tasks, directory, attempt, maximum):
        target = directory / "vbench"
        target.mkdir(parents=True, exist_ok=True)
        request = {"id": "sample", "seed": 42}
        (target / "sample.mp4").write_bytes(b"completed-video-protocol-fixture")
        commit_receipt(target, request, {"id": "sample", "artifacts": {"mp4": "sample.mp4"}},
                       rank=0, attempt_id=attempt.name)
        return dict(config=config, jobs=[dict(task=tasks[0], directory=str(directory / "vbench"),
                                              requests=[request], pending=[])],
                    output_dir=str(directory), attempt=str(attempt))

    monkeypatch.setattr(dispatcher, "make_plan", prepare)
    main(["--config", str(path), "--task-config", str(task_path), "--eval-id", "portable"])
    capsys.readouterr()
    root = Path(config["training"]["output_dir"]) / "evaluations/portable"
    Path(config["model"]["checkpoint"]).unlink()
    Path(config["data"]["validation_sources"][0]["path"]).unlink()
    scores = []

    def fake_score(runtime, task, directory):
        assert runtime is None
        scores.append(task["name"])
        return {"metric": 1.0}

    monkeypatch.setattr(scoring, "get_scorer", lambda name: fake_score)
    main(["--run-dir", str(root), "--actions", "score"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert scores == ["vbench"]


@pytest.mark.parametrize("container", ["object", "list", "jsonl"])
def test_manifest_identity_formats_and_cache_content_replacement(prepared, tmp_path, container):
    from evaluations.eval import semantic_identity
    from evaluations.artifacts import digest

    config, _, _ = prepared
    asset = tmp_path / "qwen-cache.pt"
    asset.write_bytes(b"cached-vision-a")
    record = {"task": "v2t", "caption": "blue block", "qwen_cache": asset.name,
              "video_path": "not-needed-with-cache.mp4"}
    manifest_path = tmp_path / ("records.jsonl" if container == "jsonl" else "records.json")
    manifest_path.write_text(json.dumps([record] if container == "list" else record) + "\n")
    task = dict(name="understanding", benchmark="v2t", sample={}, score={},
                data={"sources": [{"path": str(manifest_path)}]})
    before = semantic_identity(config, [task])
    assert set(before["inputs"]) == {str(asset), str(manifest_path)}
    stat = asset.stat()
    asset.write_bytes(b"cached-vision-b")
    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = semantic_identity(config, [task])
    assert digest(before) != digest(after)
    assert before["inputs"][str(manifest_path)] == after["inputs"][str(manifest_path)]


def test_t2v_identity_and_request_planning_do_not_require_target_files(prepared, tmp_path):
    from evaluations.eval import semantic_identity
    from evaluations.sampling import build_requests

    config, _, _ = prepared
    manifest_path = tmp_path / "targets.jsonl"
    manifest_path.write_text(json.dumps(dict(task="t2v", caption="blue block moves",
                                              video_path="missing-target.mp4",
                                              qwen_cache="missing-qwen.pt", wan_cache="missing-wan.pt")) + "\n")
    task = dict(name="generation", benchmark="t2v", sample={"outputs": ["latent"]}, score={},
                data={"sources": [{"path": str(manifest_path)}]})
    identity = semantic_identity(config, [task])
    assert set(identity["inputs"]) == {str(manifest_path)}
    requests = build_requests(task, config)
    assert len(requests) == 1 and requests[0]["prompt"] == "blue block moves"


def fake_completed_workers(plan, count, attempt):
    """Keep scheduler tests about orchestration; model parity has separate tests."""
    for job in plan["jobs"]:
        for request in job["pending"]:
            target = Path(job["directory"])
            name = request["id"] + ".txt"
            (target / name).write_text(request["id"])
            commit_receipt(target, request, {"id": request["id"], "artifacts": {"text": name}},
                           rank=0, attempt_id=attempt.name)


@pytest.mark.parametrize("override", ["model.lambda_text=0.9", "data.video.fps=5.0",
                                      "flow.repeats=2", "sampling.steps=2", "codec.dtype=bfloat16"])
def test_external_score_rejects_computational_overrides_without_rewriting_snapshot(
    prepared, monkeypatch, capsys, override,
):
    import evaluations.eval as dispatcher
    from evaluations.benchmarks import BENCHMARKS, Benchmark

    config, path, task_path = prepared
    monkeypatch.setitem(BENCHMARKS, "portable_protocol", Benchmark(
        sampler=lambda task, config, directory: [{"id": "one", "seed": 1}], scorer="portable_metric"))
    task_path.write_text(yaml.safe_dump(dict(name="portable", benchmark="portable_protocol", actions=["sample"],
                                           data={}, sample={}, score={})))
    monkeypatch.setattr(dispatcher, "launch", fake_completed_workers)
    main(["--config", str(path), "--task-config", str(task_path), "--eval-id", "protected"])
    capsys.readouterr()
    root = Path(config["training"]["output_dir"]) / "evaluations/protected"
    snapshot = (root / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="Offline scoring cannot change"):
        main(["--run-dir", str(root), "--actions", "score", "--set", override])
    assert (root / "config.yaml").read_bytes() == snapshot


def test_separately_selected_jobs_preserve_prior_results_and_overall_state(prepared, monkeypatch, capsys):
    import evaluations.eval as dispatcher

    config, path, first = prepared
    template = yaml.safe_load(first.read_text())
    template["data"] = {"prompts": ["blue block"]}
    first.write_text(yaml.safe_dump(dict(template, name="first")))
    second = first.with_name("second.yaml")
    second.write_text(yaml.safe_dump(dict(template, name="second")))
    monkeypatch.setattr(dispatcher, "launch", fake_completed_workers)
    main(["--config", str(path), "--task-config", str(first), "--task-config", str(second),
          "--eval-id", "selected", "--job", "first"])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "incomplete"
    first_state = deepcopy(output["tasks"]["first"])
    root = Path(config["training"]["output_dir"]) / "evaluations/selected"
    main(["--run-dir", str(root), "--job", "second", "--resume"])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["tasks"]["first"] == first_state
    assert set(output["tasks"]) == {"first", "second"}


def test_vbench_native_plan_resume_accepts_management_changes_and_identical_export_relocation(prepared, tmp_path):
    from evaluations.eval import make_plan, semantic_identity
    from evaluations.artifacts import digest

    config, _, _ = prepared
    task = dict(name="vbench", benchmark="vbench", actions=["sample", "score"], _actions=["sample"],
                data={"split": "subset"}, sample={"outputs": ["mp4"]}, score={})
    root = tmp_path / "evaluation"
    initial = make_plan(config, [task], root, tmp_path / "attempt-one", None)
    snapshot = json.loads((root / "vbench/run.json").read_text())
    original_identity = digest(semantic_identity(config, [task]))
    changed = deepcopy(config)
    changed["evaluation"].update(output_dir=str(root), num_processes=1)
    copied = tmp_path / "relocated-weights.pt"
    copied.write_bytes(Path(config["model"]["checkpoint"]).read_bytes())
    changed["model"]["checkpoint"] = str(copied)
    assert digest(semantic_identity(changed, [task])) == original_identity
    resumed = make_plan(changed, [task], root, tmp_path / "attempt-two", None)
    assert initial["jobs"][0]["requests"] == resumed["jobs"][0]["requests"]
    assert json.loads((root / "vbench/run.json").read_text()) == snapshot


def test_registered_benchmark_planner_and_scorer_dispatch_without_scheduler_changes(prepared, monkeypatch, capsys):
    import evaluations.eval as dispatcher
    from evaluations.benchmarks import BENCHMARKS, Benchmark
    from evaluations.scoring import SCORERS

    config, path, task_path = prepared
    calls = []

    def planner(task, config, directory):
        calls.append("sample")
        assert task["sample"]["protocol_version"] == 7
        return [{"id": "custom-request", "seed": 42, "task": "t2v", "prompt": "blue block"}]

    def normalize(task, config):
        calls.append("normalize")
        assert task["sample"].pop("protocol_version") == 7
        task["sample"].update(mode="t2v", outputs=["latent"], generation={"steps": 1})
        return task

    def execute(plan, count, attempt):
        calls.append("execute")
        job, = plan["jobs"]
        assert job["task"]["sample"] == {"protocol_version": 7}
        assert job["sampling_task"]["sample"] == {
            "mode": "t2v", "outputs": ["latent"], "generation": {"steps": 1}}
        assert job["sampling_task"] is not job["task"]
        fake_completed_workers(plan, count, attempt)

    def scorer(runtime, task, directory):
        calls.append("score")
        assert runtime is None
        assert task["sample"] == {"protocol_version": 7}
        assert task["score"]["metric_version"] == 3
        rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
        assert [row["id"] for row in rows] == ["custom-request"]
        return {"my_metric": len(rows)}

    monkeypatch.setitem(BENCHMARKS, "new_protocol", Benchmark(
        sampler=planner, scorer="new_metric", normalize_sample=normalize))
    monkeypatch.setitem(SCORERS, "new_metric", scorer)
    monkeypatch.setattr(dispatcher, "launch", execute)
    task_path.write_text(yaml.safe_dump(dict(name="custom", benchmark="new_protocol", actions=["sample", "score"],
                                           data={}, sample={"protocol_version": 7}, score={"metric_version": 3})))
    main(["--config", str(path), "--task-config", str(task_path), "--eval-id", "registered"])
    result = json.loads(capsys.readouterr().out)
    assert calls == ["sample", "normalize", "execute", "score"]
    assert result["status"] == "completed"
    assert result["tasks"]["custom"]["metrics"] == {"my_metric": 1}
