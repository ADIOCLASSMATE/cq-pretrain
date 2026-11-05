from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoMetadata

from configs.config import DEFAULTS, load_config, save_config, validate_config
from data.cache import load_cache
from data.dataset import VideoDataset, decode_video, read_records
from data.packing import collate_video_samples
from data.sequence import native_video_metadata
from scripts import prepare_latents
from tests.video_fixtures import make_fixture


def native_settings():
    return dict(
        mode="native", num_frames=None, fps=None, height=None, width=None,
        qwen_num_frames=None, qwen_processor={},
    )


def npy_video(tmp_path, count, height=32, width=64, fps=30000 / 1001):
    frames = np.random.default_rng(count).integers(
        0, 256, (count, height, width, 3), dtype=np.uint8
    )
    path = tmp_path / f"frames-{count}-{height}-{width}.npy"
    np.save(path, frames)
    return dict(frames_path=str(path), fps=fps), frames


@pytest.mark.parametrize("source_count,kept", [(5, 5), (6, 5), (7, 5), (8, 5), (9, 9), (84, 81), (85, 85)])
def test_native_preserves_pixels_rate_and_only_trims_tail(tmp_path, source_count, kept):
    video, frames = npy_video(tmp_path, source_count)
    rgb, clip = prepare_latents.prepare_native_video(video)
    assert rgb.dtype == torch.uint8
    np.testing.assert_array_equal(rgb.permute(0, 2, 3, 1).numpy(), frames[:kept])
    assert clip["frame_indices"] == list(range(kept))
    assert clip["num_frames"] == kept
    assert clip["source_num_frames"] == source_count
    assert (clip["height"], clip["width"], clip["fps"]) == (32, 64, video["fps"])
    np.testing.assert_array_equal(clip["timestamps"], np.arange(kept) / video["fps"])
    assert clip["spatial_transform"] == "identity"


@pytest.mark.parametrize("irregular", [False, True])
def test_container_rate_and_actual_pts_survive_native_preparation(tmp_path, irregular):
    path = tmp_path / "source.mkv"
    rate = Fraction(30000, 1001)
    ticks = [100, 101, 103, 104, 107, 109, 110, 113] if irregular else list(range(100, 108))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.height, stream.width, stream.pix_fmt = 32, 64, "bgr0"
        for index, tick in enumerate(ticks):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index * 25, np.uint8), format="rgb24")
            frame.pts, frame.time_base = tick, 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    # A stale manifest FPS must not override the actual container clock.
    video = dict(path=str(path), fps=4.0)
    frames, times, source_hash = decode_video(video)
    rgb, clip = prepare_latents.prepare_native_video(video)
    np.testing.assert_array_equal(rgb.permute(0, 2, 3, 1).numpy(), frames[:5])
    np.testing.assert_array_equal(clip["timestamps"], times[:5])
    assert clip["fps"] == float(rate)
    assert clip["source_sha256"] == source_hash
    if irregular:
        assert not np.allclose(np.diff(clip["timestamps"]), 1 / clip["fps"])


@pytest.mark.parametrize("count,height", [(4, 32), (5, 30)])
def test_native_rejects_inputs_requiring_padding_or_resize(tmp_path, count, height):
    video, _ = npy_video(tmp_path, count, height=height)
    with pytest.raises(ValueError, match="at least 5|divisible by 16"):
        prepare_latents.prepare_native_video(video)


@pytest.mark.parametrize(
    "count,sampling,expected_count",
    [(None, {"fps": 2.0, "min_frames": 4, "max_frames": 6}, 6),
     (None, {"fps": 0.1, "min_frames": 4, "max_frames": 16}, 4),
     (None, {"fps": 2.0, "min_frames": 4, "max_frames": 16}, 10),
     (7, {"fps": None}, 7)],
)
def test_qwen_uses_native_sampler_once_and_preserves_spatial_pixels(
    tmp_path, monkeypatch, count, sampling, expected_count
):
    torch.set_num_threads(2)
    video, _ = npy_video(tmp_path, 84, fps=30000 / 2002)
    rgb, clip = prepare_latents.prepare_native_video(video)
    # Deliberately nonuniform PTS: serialized times must come from PTS, not index/fps.
    clip["timestamps"] = (np.arange(81, dtype=np.float64) ** 1.01 / clip["fps"]).tolist()
    vp = Qwen3VLVideoProcessor(cap_pixels_per_frame=False)
    settings = native_settings()
    settings.update(qwen_num_frames=count, qwen_processor=sampling)
    reference = Qwen3VLVideoProcessor(
        min_frames=sampling.get("min_frames", 4),
        max_frames=sampling.get("max_frames", 768), cap_pixels_per_frame=False,
    )
    indices = reference.sample_frames(
        VideoMetadata(total_num_frames=81, fps=clip["fps"]),
        num_frames=count, fps=sampling["fps"],
    )
    expected_times = [clip["timestamps"][i] for i in indices]
    expected = reference(
        videos=[rgb[indices]], do_resize=False, do_sample_frames=False,
        video_metadata=[native_video_metadata(expected_times)], return_tensors="pt",
    )
    original_limits = (vp.min_frames, vp.max_frames)
    calls = []
    sample_frames = vp.sample_frames

    def sample(*args, **kwargs):
        calls.append((args, kwargs))
        return sample_frames(*args, **kwargs)

    monkeypatch.setattr(vp, "sample_frames", sample)
    monkeypatch.setattr(vp, "resize", lambda **kwargs: pytest.fail("Unexpected Qwen resize"))
    pixels, grid, times = prepare_latents.native_qwen_video_inputs(
        rgb, clip, SimpleNamespace(video_processor=vp), settings
    )
    assert (vp.min_frames, vp.max_frames) == original_limits
    assert len(calls) == 1
    assert len(times) == expected_count
    assert times == expected_times
    assert grid.tolist() == [(expected_count + 1) // 2, 2, 4]
    torch.testing.assert_close(pixels, expected["pixel_values_videos"], rtol=0, atol=0)
    torch.testing.assert_close(grid, expected["video_grid_thw"][0], rtol=0, atol=0)
    assert settings["qwen_processor"] == sampling


def test_qwen_rejects_duplicate_sampling_and_incompatible_geometry(tmp_path):
    video, _ = npy_video(tmp_path, 5, height=16)
    rgb, clip = prepare_latents.prepare_native_video(video)
    processor = SimpleNamespace(video_processor=Qwen3VLVideoProcessor())
    settings = native_settings()
    settings["qwen_num_frames"] = 6
    with pytest.raises(ValueError, match="exceeds the aligned clip"):
        prepare_latents.native_qwen_video_inputs(rgb, clip, processor, settings)
    settings["qwen_num_frames"] = 4
    with pytest.raises(ValueError, match="divisible by 32"):
        prepare_latents.native_qwen_video_inputs(rgb, clip, processor, settings)


def native_config():
    config = deepcopy(DEFAULTS)
    config["data"]["video"] = native_settings()
    config["data"]["representations"] = dict(qwen="cache", wan="cache")
    return config


def test_native_configuration_and_existing_defaults(tmp_path):
    config = native_config()
    validate_config(config)
    config["data"]["video"]["qwen_num_frames"] = 16
    validate_config(config)
    path = tmp_path / "native.yaml"
    save_config(config, path)
    assert load_config(path)["data"]["video"] == config["data"]["video"]
    example = Path(__file__).resolve().parents[1] / "configs/prepare_latents_native.yaml"
    assert load_config(example)["data"]["video"]["qwen_processor"]["max_frames"] == 16
    # Adding a default mode key would invalidate every existing cache's exact metadata match.
    path.write_text("{}\n")
    assert load_config(path)["data"]["video"] == DEFAULTS["data"]["video"]
    assert "mode" not in DEFAULTS["data"]["video"]


@pytest.mark.parametrize("key,value,match", [
    ("num_frames", 81, "read them from each source"),
    ("fps", 24, "read them from each source"),
    ("height", 480, "read them from each source"),
    ("width", 832, "read them from each source"),
    ("qwen_num_frames", 1, "integer >= 2"),
    ("mode", "natve", "fixed or native"),
    ("qwen_processor", {"fps": 0}, "positive and finite"),
    ("qwen_processor", {"fps": float("nan")}, "positive and finite"),
    ("qwen_processor", {"do_resize": True}, "do_resize=false"),
    ("qwen_processor", {"min_frames": 8, "max_frames": 4}, "min_frames <= max_frames"),
    ("qwen_processor", {"num_frames": 8}, "preparation owns sampling"),
    ("qwen_processor", {"do_sample_frames": True}, "preparation owns sampling"),
])
def test_native_configuration_rejects_conflicting_parameters(key, value, match):
    config = native_config()
    config["data"]["video"][key] = value
    with pytest.raises(ValueError, match=match):
        validate_config(config)


@pytest.mark.parametrize("kind", ["qwen", "wan"])
@pytest.mark.parametrize("mode", ["online", "prefer_cache"])
def test_native_configuration_supports_shared_online_preprocessing(kind, mode):
    config = native_config()
    config["data"]["representations"][kind] = mode
    validate_config(config)
    assert config["data"]["video"]["mode"] == "native"


def test_native_qwen_fps_and_frame_count_are_mutually_exclusive():
    config = native_config()
    config["data"]["video"].update(qwen_num_frames=8, qwen_processor={"fps": 2.0})
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(config)


def test_prepare_native_caches_load_without_changing_dataset(tmp_path, monkeypatch):
    from data.codec import WanVideoCodec
    from models.vision_encoder import Qwen35VideoEncoder

    torch.set_num_threads(2)
    config, processor, model, _, _ = make_fixture(tmp_path / "fixture")
    config["data"]["video"] = native_settings()
    config["data"]["video"].update(qwen_num_frames=3, qwen_processor={"fps": None})
    inputs = [npy_video(tmp_path, 11, 16, 32), npy_video(tmp_path, 14, 32, 16, fps=12.5)]
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(dict(videos=[video], caption="blue block moves")) + "\n" for video, _ in inputs))
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    encoded = []
    encoder_loads = []

    class Codec:
        # This test verifies the exact VAE input and cache interface, not VAE numerics.
        def encode(self, rgb):
            encoded.append(rgb.clone())
            return torch.zeros(16, (rgb.shape[1] - 1) // 4 + 1, rgb.shape[2] // 8, rgb.shape[3] // 8)

    def load_encoder(*args, **kwargs):
        encoder_loads.append(True)
        return model.vision_encoder

    monkeypatch.setattr(WanVideoCodec, "from_pretrained", lambda *args, **kwargs: Codec())
    monkeypatch.setattr(Qwen35VideoEncoder, "from_pretrained", load_encoder)
    args = ["--config", str(config_path), "--manifest", str(manifest), "--task", "t2v",
            "--kind", "both", "--cache-dir", str(tmp_path / "cache")]
    output = tmp_path / "cached.jsonl"
    prepare_latents.main(args + ["--output", str(output)])
    assert len(encoded) == 2 and len(encoder_loads) == 1
    records = read_records(dict(path=str(output)))
    for i, ((video, frames), kept) in enumerate(zip(inputs, [9, 13])):
        expected = torch.from_numpy(frames[:kept]).permute(3, 0, 1, 2).float().div(127.5).sub(1)
        torch.testing.assert_close(encoded[i], expected, rtol=0, atol=0)
        qwen = load_cache(records[i]["videos"][0]["qwen_cache"], "qwen", preprocessing=config["data"]["video"])
        wan = load_cache(records[i]["videos"][0]["wan_cache"], "wan", preprocessing=config["data"]["video"])
        assert qwen["metadata"]["clip"] == wan["metadata"]["clip"]
        assert qwen["metadata"]["qwen_timestamps"] == [j / video["fps"] for j in [0, (kept - 1) // 2, kept - 1]]
        assert qwen["features"].shape[0] == int(qwen["grid_thw"].prod()) // 4
    # Cache reuse must skip both encoders.
    prepare_latents.main(args + ["--output", str(tmp_path / "reused.jsonl")])
    assert len(encoded) == 2 and len(encoder_loads) == 1
    monkeypatch.setattr("data.dataset.prepare_raw_video", lambda *args: pytest.fail("Unexpected online preprocessing"))
    dataset = VideoDataset(dict(path=str(output)), processor, config)
    batch = collate_video_samples([dataset[0], dataset[1]])
    assert batch["wan_latents"][0].shape == (16, 3, 2, 4)
    assert batch["wan_latents"][1].shape == (16, 4, 4, 2)
    with torch.no_grad():
        prepared = model.prepare_inputs(batch)
    assert prepared["inputs_embeds"].shape[0] == 2
    # Missing caches fail instead of silently changing the source video.
    missing = deepcopy(records)
    missing[0]["videos"][0].pop("wan_cache")
    with pytest.raises(ValueError, match="wan cache required"):
        VideoDataset(dict(path=str(output)), processor, config, records=missing)[0]
    changed = deepcopy(config["data"]["video"])
    changed["qwen_num_frames"] = 4
    with pytest.raises(ValueError, match="cache preprocessing differs"):
        load_cache(records[0]["videos"][0]["qwen_cache"], "qwen", preprocessing=changed)
