from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from configs.config import load_config, save_config
from data.cache import load_cache, branch_preprocessing
from data.dataset import VideoDataset, read_records
from data.manifest import read_manifest, original_digest, merge_latent_updates
from data.video import prepare_video
from scripts.prepare_latents import main
from tests.video_fixtures import make_fixture


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_in_place_precision_interruption_resume_and_reading(
    tmp_path, monkeypatch, dtype
):
    from data.codec import WanVideoCodec
    from models.backbone_qwen35 import Qwen35VideoEncoder

    config, processor, model, _, _ = make_fixture(tmp_path / "fixture")
    config["training"]["mixed_precision"] = "bf16" if dtype == torch.bfloat16 else "no"
    config["data"]["max_seq_length"] = 2048
    config["data"]["video"] = dict(
        mode="native",
        num_frames=None,
        fps=None,
        height=None,
        width=None,
        expected_num_frames=9,
        qwen_num_frames=3,
        qwen_processor={"fps": None, "do_resize": False},
    )
    rows = []
    for i, (h, w) in enumerate(((16, 32), (32, 16))):
        path = tmp_path / f"raw{i}.npy"
        np.save(path, np.random.default_rng(i).integers(0, 256, (9, h, w, 3), np.uint8))
        rows.append(
            dict(
                sample_id=f"sample-{i}",
                frames_path=path.name,
                caption="blue block moves",
                fps=12.5,
                num_frames=9,
                height=h,
                width=w,
                source={"untouched": [i, "value"]},
            )
        )
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    calls = {"qwen": 0, "wan": 0}

    def qwen_loader(*args, **kwargs):
        calls["qwen"] += 1
        return model.vision_encoder.to(dtype=dtype)

    class Codec:
        def encode(self, rgb):
            calls["wan"] += 1
            return torch.zeros(16, 3, rgb.shape[2] // 8, rgb.shape[3] // 8)

    monkeypatch.setattr(Qwen35VideoEncoder, "from_pretrained", qwen_loader)
    monkeypatch.setattr(WanVideoCodec, "from_pretrained", lambda *a, **k: Codec())
    args = [
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--in-place",
        "--flush-every",
        "1",
    ]
    # A completed Qwen file survives interruption before the first index commit.
    with pytest.raises(RuntimeError, match="Intentional interruption"):
        main(args + ["--stop-after-files", "1"])
    assert read_manifest(manifest) == rows
    saved = list((tmp_path / "latents").rglob("*.pt"))
    assert len(saved) == 1
    prior_mtime = saved[0].stat().st_mtime_ns
    report = main(args + ["--verify-online"])
    assert report["complete"] and report["reused"]["qwen"] == 1
    assert saved[0].stat().st_mtime_ns == prior_mtime
    updated = read_manifest(manifest)
    for before, after in zip(rows, updated):
        assert original_digest(before) == original_digest(after)
        assert not Path(after["qwen_cache"]).is_absolute()
        assert after["latent_info"]["qwen"]["dtype"] == str(dtype).split(".")[1]
    # Full resume must neither load an encoder nor rewrite latent files.
    mtimes = {p: p.stat().st_mtime_ns for p in (tmp_path / "latents").rglob("*.pt")}
    calls_before = calls.copy()
    report = main(args + ["--verify-only"])
    assert report["encoded"] == {"qwen": 0, "wan": 0}
    assert calls == calls_before
    assert mtimes == {p: p.stat().st_mtime_ns for p in mtimes}
    # A training config only needs the original manifest and the task to infer native settings.
    values = yaml.safe_load(config_path.read_text())
    del values["data"]["video"]
    values["data"]["sources"] = [{"path": str(manifest), "task": "t2v"}]
    inference_path = tmp_path / "training.yaml"
    inference_path.write_text(yaml.safe_dump(values))
    inferred = load_config(inference_path)
    assert inferred["data"]["video"] == config["data"]["video"]
    source = inferred["data"]["sources"][0]
    for modes in (
        {"qwen": "cache", "wan": "cache"},
        {"qwen": "online", "wan": "cache"},
        {"qwen": "cache", "wan": "online"},
        {"qwen": "online", "wan": "online"},
    ):
        inferred["data"]["representations"] = modes
        item = VideoDataset(source, processor, inferred)[0]
        assert item["clip"]["num_frames"] == 9
    # Changing only Qwen sampling leaves an independently cached Wan latent valid.
    inferred["data"]["representations"] = {"qwen": "online", "wan": "cache"}
    inferred["data"]["video"]["qwen_num_frames"] = 5
    assert VideoDataset(source, processor, inferred)[0]["wan_latent"] is not None
    inferred["data"]["representations"]["qwen"] = "cache"
    with pytest.raises(ValueError, match="preprocessing differs"):
        VideoDataset(source, processor, inferred)[0]
    # Bind each tensor to its sample, not just its shape/checkpoint.
    normalized = read_records(source)
    normalized[0]["videos"][0]["qwen_cache"] = normalized[1]["videos"][0]["qwen_cache"]
    with pytest.raises(ValueError, match="sample_id differs"):
        VideoDataset(source, processor, config, records=normalized)[0]
    # Byte corruption is diagnosed on resume without silently replacing artifacts.
    qpath = tmp_path / updated[0]["qwen_cache"]
    with qpath.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum differs"):
        main(args + ["--verify-only"])


def test_exact_native_frame_count_is_not_silently_trimmed(tmp_path):
    path = tmp_path / "raw.npy"
    np.save(path, np.zeros((84, 32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="differs from expected 81"):
        prepare_video(
            {"frames_path": str(path), "fps": 15.0, "num_frames": 81},
            {"mode": "native"},
        )


def test_index_merge_preserves_other_worker_and_rejects_source_edits(tmp_path):
    path = tmp_path / "samples.jsonl"
    rows = [{"sample_id": str(i), "caption": str(i)} for i in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    originals = {r["sample_id"]: original_digest(r) for r in rows}
    merge_latent_updates(path, {"0": {"qwen_cache": "q.pt"}}, originals)
    merge_latent_updates(path, {"1": {"wan_cache": "w.pt"}}, originals)
    result = read_manifest(path)
    assert result[0]["qwen_cache"] == "q.pt" and result[1]["wan_cache"] == "w.pt"
    result[2]["caption"] = "changed"
    path.write_text("".join(json.dumps(r) + "\n" for r in result))
    with pytest.raises(ValueError, match="Original sample fields changed"):
        merge_latent_updates(path, {}, originals)
