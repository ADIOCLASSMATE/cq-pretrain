"""Sampling adapter parity and condition/artifact isolation."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluations.sampling import (
    artifact_paths, build_requests, sample_request, video_layout,
)
from evaluations.validation import seed_evaluation
from data.dataset import VideoDataset
from data.packing import collate_video_samples
from tests.video_fixtures import make_fixture


@pytest.fixture
def runtime(tmp_path):
    torch.set_num_threads(2)
    config, processor, model, _, _ = make_fixture(tmp_path / "fixture")
    config["sampling"].update(steps=2, cfg=1.0, max_new_tokens=3, temperature=0.7)
    return SimpleNamespace(config=config, processor=processor, get_model=lambda: model,
                           device=torch.device("cpu"), autocast=nullcontext, model=model)


def test_plan_does_not_read_t2v_target_and_stable_shards(runtime, tmp_path):
    manifest = tmp_path / "t2v.jsonl"
    manifest.write_text(json.dumps({"sample_id": "odd / id", "caption": "blue block moves",
                                    "video_path": "/missing/target.mp4", "wan_cache": "/missing/z.pt"}) + "\n")
    task = dict(benchmark="t2v", data={"sources": [{"path": str(manifest)}]},
                sample={"samples_per_input": 5, "outputs": ["latent"]})
    requests = build_requests(task, runtime.config)
    assert len(requests) == 5
    assert len({row["id"] for row in requests}) == 5
    assert all("/" not in row["id"] for row in requests)
    assert sorted(row["seed"] for rank in range(4) for row in requests[rank::4]) == [
        runtime.config["sampling"]["seed"] + i for i in range(5)
    ]
    # Existence and content of targets do not enter the sampling request condition.
    assert requests[0]["prompt"] == "blue block moves"
    output = sample_request(runtime, task, requests[0], tmp_path / "out")
    assert (tmp_path / "out" / output["artifacts"]["latent"]).is_file()


def test_t2v_native_generation_parity_and_collection_does_not_change_output(runtime, tmp_path):
    task = dict(benchmark="t2v", data={"prompt": "blue block moves"},
                sample={"outputs": ["latent"], "seed": 1234})
    request = build_requests(task, runtime.config)[0]
    model, config = runtime.model, runtime.config
    batch = video_layout(request["prompt"], runtime.processor, config)
    video, sampling = config["data"]["video"], config["sampling"]
    seed_evaluation(request["seed"])
    rng = torch.Generator(device="cpu").manual_seed(request["seed"])
    with torch.no_grad():
        expected = model.generate_video(
            batch, latent_shapes=[(16, (video["num_frames"] - 1) // 4 + 1,
                                   video["height"] // 8, video["width"] // 8)],
            steps=sampling["steps"], solver=sampling["solver"], cfg=sampling["cfg"],
            time_shift=sampling["time_shift"], timestep_scale=config["flow"]["timestep_scale"], rng=rng,
        )[0]
    model.train()
    rng_before = torch.get_rng_state().clone()
    record = sample_request(runtime, task, request, tmp_path / "normal")
    actual = torch.load(tmp_path / "normal" / record["artifacts"]["latent"], weights_only=True)["latent"]
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert model.training
    torch.testing.assert_close(rng_before, torch.get_rng_state(), atol=0, rtol=0)
    collected = deepcopy(task)
    collected["sample"]["collect"] = {"initial_noise": True}
    record = sample_request(runtime, collected, request, tmp_path / "collected")
    actual = torch.load(tmp_path / "collected" / record["artifacts"]["latent"], weights_only=True)["latent"]
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert set(record["artifacts"]) == {"initial_noise", "latent"}
    assert not list((tmp_path / "collected").rglob("*.tmp*"))


def test_v2t_native_tokens_logits_parity_and_answer_isolation(runtime, tmp_path):
    source = deepcopy(runtime.config["data"]["sources"][0])
    rows = [json.loads(line) for line in Path(source["path"]).read_text().splitlines()]
    manifest = tmp_path / "v2t.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows if row["task"] == "v2t") + "\n")
    source["path"] = str(manifest)
    task = dict(benchmark="v2t", data={"sources": [source]},
                sample={"outputs": ["text"], "collect": {"token_ids": True, "text_logits": True}})
    request = build_requests(task, runtime.config)[0]
    p, config = runtime.processor, runtime.config
    batch = collate_video_samples([VideoDataset(source, p, config, prompt_only=True)[0]],
                                  pad_token_id=p.tokenizer.pad_token_id or 0)
    seed_evaluation(request["seed"])
    expected = runtime.model.generate_text(
        batch, max_new_tokens=config["sampling"]["max_new_tokens"],
        temperature=config["sampling"]["temperature"], eos_token_id=p.tokenizer.eos_token_id,
        use_cache=False, return_logits=True,
    )
    record = sample_request(runtime, task, request, tmp_path / "actual")
    actual = torch.load(tmp_path / "actual" / record["artifacts"]["text_logits"], weights_only=True)
    torch.testing.assert_close(expected["sequences"][0], actual["token_ids"], atol=0, rtol=0)
    torch.testing.assert_close(expected["logits"][0], actual["logits"], atol=0, rtol=0)
    raw = [json.loads(line) for line in Path(source["path"]).read_text().splitlines()]
    raw[0]["caption"] = "red red red red red"
    Path(source["path"]).write_text("\n".join(map(json.dumps, raw)) + "\n")
    record = sample_request(runtime, task, request, tmp_path / "changed_answer")
    changed = torch.load(tmp_path / "changed_answer" / record["artifacts"]["text_logits"], weights_only=True)
    torch.testing.assert_close(actual["token_ids"], changed["token_ids"], atol=0, rtol=0)
    torch.testing.assert_close(actual["logits"], changed["logits"], atol=0, rtol=0)


def test_artifact_requirements_paths_and_unsupported_trajectory(runtime, tmp_path):
    task = dict(benchmark="t2v", data={"prompts": ["blue block", "red block"]},
                sample={"outputs": ["mp4", "latent"], "collect": {"token_ids": False}})
    requests = build_requests(task, runtime.config)
    official = dict(requests[0], path="videos/dimension/blue block-0.mp4")
    assert artifact_paths(task, official)["mp4"] == official["path"]
    with pytest.raises(ValueError, match="inside"):
        sample_request(runtime, task, dict(official, path="../escape.mp4"), tmp_path)
    task["sample"]["collect"]["latent_steps"] = [1]
    with pytest.raises(ValueError, match="trajectories"):
        build_requests(task, runtime.config)


def test_runtime_lazy_loads_once_and_preserves_construction_rng(runtime, monkeypatch):
    import evaluations.runtime as module

    calls = []
    monkeypatch.setattr(module, "prepare_runtime", lambda config: (torch.device("cpu"), None))
    monkeypatch.setattr(module, "load_processor", lambda config: runtime.processor)
    def load(config):
        calls.append("model")
        torch.rand(12)
        return runtime.model
    monkeypatch.setattr(module, "load_model", load)
    instance = module.EvaluationRuntime(runtime.config)
    assert not calls
    state = torch.get_rng_state().clone()
    assert instance.get_model() is instance.get_model() is runtime.model
    assert calls == ["model"]
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)


def test_inherited_mixed_sources_filter_without_changing_indices(runtime):
    runtime.config["data"]["validation_sources"] = runtime.config["data"]["sources"]
    requests = build_requests({"benchmark": "v2t", "sample": {"seed": None}}, runtime.config)
    assert [row["index"] for row in requests] == [2, 3]
    assert [row["task"] for row in requests] == ["v2t", "v2t"]
    assert requests[0]["seed"] == runtime.config["sampling"]["seed"]
    with pytest.raises(ValueError, match="Unknown sample.generation"):
        build_requests({"benchmark": "v2t", "sample": {"generation": {"stepps": 1}}}, runtime.config)


def test_training_validation_verification_helper_cpu(runtime, tmp_path):
    from tests.validation.verify_unified_evaluation import validate_loss_parity, validate_sampling_parity

    runtime.config["data"]["validation_sources"] = runtime.config["data"]["sources"]
    runtime.codec = None  # All fixture Wan latents are cached.
    result = validate_loss_parity(runtime, tmp_path)
    assert result["metrics_equal"] and result["batches_equal"]
    assert result["batches"] == 4
    parity = validate_sampling_parity(runtime, tmp_path / "sampling")
    assert parity["t2v"]["equal"]
    assert parity["v2t"]["tokens"]["equal"] and parity["v2t"]["logits"]["equal"]


def test_codec_construction_preserves_rng(runtime, monkeypatch):
    from evaluations.runtime import EvaluationVideoCodec
    from data.codec import WanVideoCodec

    sentinel = object()
    def create(*args, **kwargs):
        torch.rand(50)
        return sentinel
    monkeypatch.setattr(WanVideoCodec, "from_pretrained", create)
    state = torch.get_rng_state().clone()
    codec = EvaluationVideoCodec(runtime.config, "cpu")
    assert codec.get() is codec.get() is sentinel
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
