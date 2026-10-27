"""Benchmark data, reproducible generation, and official-process integration contracts."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from evaluations.benchmarks.vbench import generate as gen, score


def test_official_metadata_and_subset_preserve_annotations_and_shared_groups():
    full, digest = gen.load_prompts("full")
    subset, _ = gen.load_prompts("subset")
    assert digest == gen.FULL_SHA256
    assert len(full) == 946
    assert len({row["prompt_en"] for row in full}) == 944
    groups = Counter(row["dimension"][0] for row in subset)
    assert len(subset) == 22 and len(groups) == 11 and set(groups.values()) == {2}
    assert {d for row in subset for d in row["dimension"]} == {
        d for row in full for d in row["dimension"]
    }
    assert all(row in full for row in subset)
    assert len(gen.make_tasks(subset, 42)) == 150
    assert len(gen.make_tasks(full, 42)) == 6230
    assert len({task["path"] for task in gen.make_tasks(full, 42)}) == 6230


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_sharding_has_no_padding_and_seeds_survive_subset_resume_and_supplement(world_size):
    full, _ = gen.load_prompts("full")
    subset, _ = gen.load_prompts("subset")
    tasks = gen.make_tasks(subset, 123)
    shards = [gen.shard_tasks(tasks, rank, world_size) for rank in range(world_size)]
    actual = [task for shard in shards for task in shard]
    assert len(actual) == len(tasks)
    assert {t["path"] for t in actual} == {t["path"] for t in tasks}
    full_seeds = {t["path"]: t["seed"] for t in gen.make_tasks(full, 123)}
    supplemented = {t["path"]: t["seed"] for t in gen.make_tasks(subset, 123, 50)}
    assert all(full_seeds[t["path"]] == supplemented[t["path"]] == t["seed"] for t in actual)
    assert len(shards[0]) - len(shards[-1]) <= 1


def make_run(tmp_path, mode="subset"):
    directory = tmp_path / "run with spaces"
    rows, digest = gen.load_prompts(mode)
    run = {"schema_version": 1, "mode": mode, "vbench_commit": gen.VBENCH_COMMIT,
           "full_sha256": gen.FULL_SHA256, "prompts_sha256": digest,
           "config": {"sampling": {"seed": 42}, "data": {"video": {
               "width": 16, "height": 16, "num_frames": 5, "fps": 4.0,
           }}}, "checkpoint": {"path": "/not/needed/on/scoring/host.pt", "size": 123, "mtime_ns": 123},
           "seed_rule": "sampling.seed + video_index", "flicker_candidates": 25}
    gen.prepare_run(directory, run, (gen.DATA_DIR / f"{mode}.json").read_bytes())
    return directory, run, rows


def test_resume_checks_identity_and_allows_only_candidate_budget_growth(tmp_path):
    directory, run, rows = make_run(tmp_path)
    raw = (gen.DATA_DIR / "subset.json").read_bytes()
    changed = deepcopy(run)
    changed["checkpoint"]["mtime_ns"] += 1
    with pytest.raises(ValueError, match="different checkpoint"):
        gen.prepare_run(directory, changed, raw)
    changed = deepcopy(run)
    changed["flicker_candidates"] = 50
    gen.prepare_run(directory, changed, raw)
    assert score.read_run(directory)[0]["flicker_candidates"] == 50
    with pytest.raises(ValueError, match=">= 50"):
        gen.prepare_run(directory, run, raw)
    (directory / "prompts.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="prompt list differs"):
        score.read_run(directory)


def test_umm_generation_writes_mp4_and_resumes_without_loading_model(tmp_path, monkeypatch):
    import torch
    import torch.nn.functional as F
    import yaml
    from data.codec import WanVideoCodec
    from tests.video_fixtures import make_fixture
    from evaluations import runtime as evaluation_runtime, sampling as common_sampling
    from utils.checkpoint import export_weights

    torch.set_num_threads(1)
    config, _, model, _, _ = make_fixture(tmp_path / "fixture")
    config["sampling"].update(steps=2, cfg=1.0)
    export = export_weights(tmp_path / "weights.pt", model=model, config=config)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    codec_loads = []
    sample_requests = []
    original_sample = common_sampling.sample_request

    def sample(runtime, task, request, output_dir):
        sample_requests.append(dict(request))
        return original_sample(runtime, task, request, output_dir)

    monkeypatch.setattr(common_sampling, "sample_request", sample)

    class CodecStub:
        # Real UMM generation and video encoding; isolate heavyweight pretrained VAE weights.
        def decode(self, latents):
            return [F.interpolate(z[:3][None], size=(5, 16, 16), mode="trilinear")[0].tanh()
                    for z in latents]

    def codec_factory(*args, **kwargs):
        codec_loads.append(1)
        return CodecStub()

    monkeypatch.setattr(WanVideoCodec, "from_pretrained", codec_factory)
    output = tmp_path / "generated"
    arguments = ["--config", str(config_path), "--checkpoint", str(export),
                 "--output", str(output), "--mode", "subset", "--max-videos", "2"]
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    gen.main(arguments)
    assert len(codec_loads) == 1
    videos = sorted((output / "videos").rglob("*.mp4"))
    assert len(videos) == 2
    assert all(gen.video_is_complete(p, config["data"]["video"]) for p in videos)
    original = videos[0].read_bytes()
    records = [json.loads(s) for s in (output / "generation.rank-0.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in records] == [42, 43]
    with monkeypatch.context() as context:
        context.setattr(evaluation_runtime, "load_model", lambda *_: pytest.fail("Completed run reloaded model"))
        gen.main(arguments)
    videos[0].write_bytes(b"interrupted file")
    gen.main(arguments)
    assert videos[0].read_bytes() == original
    assert len(codec_loads) == 2
    assert [request["seed"] for request in sample_requests] == [42, 43, 42]
    assert all(request["task"] == "t2v" for request in sample_requests)
    assert sample_requests[0]["path"] == str(videos[0].relative_to(output))
    assert not list((output / ".staging").iterdir())
    with pytest.raises(ValueError, match="different checkpoint/config"):
        gen.main(arguments + ["--set", "sampling.steps=3"])


def fake_official(monkeypatch, *, failure=None):
    full, _ = gen.load_prompts("full")
    mapping = {dimension: row["dimension"][0] for row in full for dimension in row["dimension"]}
    monkeypatch.setattr(score, "official_checkout", lambda _: mapping)
    monkeypatch.setattr(score, "require_cuda", lambda: None)
    monkeypatch.setattr(score, "video_is_complete", lambda p, _: not (
        failure == "static" and "filtered_videos" in p.parts
    ))
    calls = []

    def execute(command, root, log):
        calls.append(command)
        log.write_text("fake subprocess output\n")
        if Path(command[1]).name == "evaluate.py":
            dimension = command[command.index("--dimension") + 1]
            output = Path(command[command.index("--output_path") + 1])
            videos = Path(command[command.index("--videos_path") + 1])
            metadata = Path(command[command.index("--full_json_dir") + 1])
            selected = [row for row in json.loads(metadata.read_text()) if dimension in row["dimension"]]
            for row in selected:
                row["video_list"] = [str(videos / f"{row['prompt_en']}-{i}.mp4") for i in range(5)]
            value = float("nan") if failure == "nan" and dimension == "color" else 0.5
            payload = {} if failure == "missing" and dimension == "color" else {dimension: [value, []]}
            if failure == "video" and dimension == "color":
                selected[0]["video_list"].pop()
            (output / "test_eval_results.json").write_text(json.dumps(payload))
            (output / "test_full_info.json").write_text(json.dumps(selected))
        elif Path(command[1]).name == "cal_final_score.py":
            archive = Path(command[command.index("--zip_file") + 1])
            with zipfile.ZipFile(archive) as z:
                assert set(z.namelist()) == {f"{d}.json" for d in mapping}
                assert all(set(json.loads(z.read(f"{d}.json"))) == {d} for d in mapping)
            log.write_text("| quality score|0.1|\n| semantic score|0.2|\n| total score|0.12|\n")

    monkeypatch.setattr(score, "run_official", execute)
    return calls


@pytest.mark.parametrize("mode", ["subset", "full"])
def test_official_command_contract_and_full_only_aggregation(tmp_path, monkeypatch, mode):
    directory, _, rows = make_run(tmp_path, mode)
    calls = fake_official(monkeypatch)
    score.score_run(directory, tmp_path / "official checkout")
    summary_path, = (directory / "scores").glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert len(summary["dimensions"]) == 16
    assert summary["scoring_videos"] == len(rows) * 5
    assert ("aggregate" in summary) == (mode == "full")
    assert len([c for c in calls if Path(c[1]).name == "evaluate.py"]) == 16
    assert len([c for c in calls if Path(c[1]).name == "cal_final_score.py"]) == int(mode == "full")
    assert all(c[c.index("--mode") + 1] == "vbench_standard" for c in calls if "--mode" in c)
    assert all(c[c.index("--full_json_dir") + 1] == str(directory / "prompts.json")
               for c in calls if "--full_json_dir" in c)


@pytest.mark.parametrize("failure", ["static", "missing", "nan", "video"])
def test_incomplete_or_nonfinite_official_results_never_produce_a_total(tmp_path, monkeypatch, failure):
    directory, _, _ = make_run(tmp_path, "full")
    calls = fake_official(monkeypatch, failure=failure)
    with pytest.raises(ValueError):
        score.score_run(directory, tmp_path / "official")
    assert not list((directory / "scores").glob("*/summary.json"))
    assert not any(Path(c[1]).name == "cal_final_score.py" for c in calls)


def test_score_dry_run_does_not_require_cuda_or_create_results(tmp_path, monkeypatch):
    directory, _, _ = make_run(tmp_path)
    calls = fake_official(monkeypatch)
    monkeypatch.setattr(score, "require_cuda", lambda: pytest.fail("dry-run initialized CUDA"))
    score.score_run(directory, tmp_path / "official", dry_run=True)
    assert not calls
    assert not (directory / "scores").exists()


def test_no_silent_npu_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NPU scoring has not been ported"):
        score.require_cuda()


def test_official_subprocess_isolated_and_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("PYTHONPATH", "/not/the/official/checkout")
    code = ("import os,sys; assert 'RANK' not in os.environ; assert 'WORLD_SIZE' not in os.environ; "
            "assert os.environ['PYTHONPATH']==os.getcwd(); print('official error',flush=True); sys.exit(7)")
    with pytest.raises(RuntimeError, match="exited 7.*log:"):
        score.run_official([sys.executable, "-c", code], tmp_path, tmp_path / "error.log")
    assert "official error" in (tmp_path / "error.log").read_text()


@pytest.mark.parametrize("kind", ["generate", "score"])
def test_shell_launchers_quote_paths_and_propagate_exit(tmp_path, kind):
    fake_python = tmp_path / "python with spaces"
    fake_python.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(17)\n")
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHON_BIN=str(fake_python), VBENCH_PYTHON=str(fake_python),
               UMM_DEVICE="cpu", NPROC_PER_NODE="4")
    result = subprocess.run(
        ["bash", str(root / f"evaluations/scripts/vbench_{kind}.sh"), "--output", "a path with spaces"],
        cwd=tmp_path, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 17, result.stderr
    args = json.loads(result.stdout)
    assert args[-2:] == ["--output", "a path with spaces"]
    assert f"evaluations.benchmarks.vbench.{kind}" in args
    if kind == "generate":
        assert "--nproc_per_node=4" in args


def test_official_full_hash_is_checked_before_use(tmp_path, monkeypatch):
    raw = (gen.DATA_DIR / "full.json").read_bytes()
    (tmp_path / "full.json").write_bytes(raw + b"\n")
    monkeypatch.setattr(gen, "DATA_DIR", tmp_path)
    assert hashlib.sha256(raw).hexdigest() == gen.FULL_SHA256
    with pytest.raises(ValueError, match="pinned official"):
        gen.load_prompts("full")


def test_scoring_rejects_unpinned_or_modified_official_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(score.subprocess, "check_output", lambda *a, **k: "different commit\n")
    with pytest.raises(ValueError, match="Use official Vchitect/VBench commit"):
        score.official_checkout(tmp_path)
    monkeypatch.setattr(score.subprocess, "check_output", lambda *a, **k: gen.VBENCH_COMMIT + "\n")

    def dirty(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(score.subprocess, "run", dirty)
    with pytest.raises(subprocess.CalledProcessError):
        score.official_checkout(tmp_path)
