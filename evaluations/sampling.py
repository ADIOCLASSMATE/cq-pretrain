"""Task-driven T2V/V2T sampling and atomic final-artifact collection.

Request plans are serializable and contain no decoded target video or features.
The dispatcher owns sharding/resume; this module owns one request's computation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

import torch

from data.dataset import VideoDataset, qwen_video_inputs, read_records
from data.packing import collate_video_samples
from data.sequence import make_video_sequence
from evaluations.validation import evaluation_state, seed_evaluation
from training.loss import to_device


SEED_POLICY = "per_request_v1"


def video_layout(prompt, processor, config, *, messages=None):
    """Native geometry from zero frames; target pixels/features never enter CQ."""
    settings = config["data"]["video"]
    rgb = torch.zeros(settings["num_frames"], 3, settings["height"], settings["width"], dtype=torch.uint8)
    clip = {"timestamps": [i / settings["fps"] for i in range(settings["num_frames"])]}
    _, grid, times = qwen_video_inputs(rgb, clip, processor, settings)
    if messages is None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "video", "video_id": "v0"}]},
        ]
    else:
        messages = _target_layout_messages(messages)
    sequence = make_video_sequence(
        messages, processor, grid, times,
        max_seq_length=config["data"]["max_seq_length"],
    )
    row = dict(sequence, task="t2v", sample_id="generation", clip=clip,
               video_features=None, pixel_values_videos=None, video_grid_thw=grid)
    batch = collate_video_samples([row], pad_token_id=processor.tokenizer.pad_token_id or 0)
    batch["video_observed_mask"].fill_(False)
    return batch


def _target_layout_messages(messages):
    """Keep conditioning turns and the target placeholder, never target suffix text."""
    result = deepcopy(messages)
    visual = [(i, j) for i, message in enumerate(result)
              for j, part in enumerate(message["content"]) if part["type"] == "video"]
    if len(visual) != 1 or result[visual[0][0]]["role"] != "assistant":
        raise ValueError("T2V sampling needs exactly one assistant video target")
    i, j = visual[0]
    # Text preceding the placeholder is a known assistant prefix; all following
    # text is a target suffix and cannot become a generation condition.
    result[i]["content"] = result[i]["content"][:j + 1]
    return result[:i + 1]


def write_video(path, video, fps):
    import av
    from fractions import Fraction

    frames = video.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).cpu().numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _safe_id(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._")[:80] or "sample"


def _prompt(messages):
    parts = []
    for message in messages:
        if any(part["type"] == "video" for part in message["content"]):
            if message["role"] == "assistant":
                break
        parts.extend(part["text"] for part in message["content"] if part["type"] == "text")
    return "\n".join(parts)


def validate_sampling_task(task):
    """Validate the common generator adapter before loading models or artifacts."""
    sample = task.get("sample") or {}
    allowed = {"mode", "generation", "video", "seed", "max_samples", "samples_per_input",
               "samples_per_prompt", "outputs", "collect", "flicker_candidates"}
    if set(sample) - allowed:
        raise ValueError(f"Unknown sample parameters: {sorted(set(sample) - allowed)}")
    collect = sample.get("collect") or {}
    permitted = {"token_ids", "text_logits", "final_latent", "initial_noise", "latent_steps"}
    if set(collect) - permitted:
        raise ValueError(f"Unknown sample.collect parameters: {sorted(set(collect) - permitted)}")
    if collect.get("latent_steps"):
        raise ValueError("Intermediate sampling trajectories are not supported; models are unchanged")
    if any(type(value) is not bool for key, value in collect.items() if key != "latent_steps"):
        raise ValueError("sample.collect flags must be booleans")
    generation = sample.get("generation") or {}
    allowed_generation = {"steps", "solver", "cfg", "time_shift", "max_new_tokens",
                          "temperature", "top_k", "top_p"}
    if set(generation) - allowed_generation:
        raise ValueError(f"Unknown sample.generation parameters: {sorted(set(generation) - allowed_generation)}")
    for key, minimum in (("steps", 1), ("max_new_tokens", 0), ("top_k", 0)):
        if key in generation and (type(generation[key]) is not int or generation[key] < minimum):
            raise ValueError(f"sample.generation.{key} must be an integer >= {minimum}")
    if generation.get("solver", "euler") not in {"euler", "heun"}:
        raise ValueError("sample.generation.solver must be euler or heun")
    for key in ("cfg", "time_shift", "temperature", "top_p"):
        if key in generation and (isinstance(generation[key], bool)
                                  or not math.isfinite(float(generation[key]))):
            raise ValueError(f"sample.generation.{key} must be finite")
    if float(generation.get("time_shift", 1)) <= 0 or not 0 < float(generation.get("top_p", 1)) <= 1:
        raise ValueError("time_shift must be positive and top_p must be in (0, 1]")
    video = sample.get("video") or {}
    video_keys = {"num_frames", "fps", "height", "width", "qwen_num_frames", "qwen_processor"}
    if set(video) - video_keys:
        raise ValueError(f"Unknown sample.video parameters: {sorted(set(video) - video_keys)}")
    for key in ("num_frames", "height", "width", "qwen_num_frames"):
        value = video.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"sample.video.{key} must be a positive integer")
    if "num_frames" in video and (video["num_frames"] - 1) % 4:
        raise ValueError("Wan sampling requires 4k+1 video frames")
    if any(video.get(key, 16) % 16 for key in ("height", "width")):
        raise ValueError("Wan video height/width must be divisible by 16")
    if "fps" in video and (not math.isfinite(float(video["fps"])) or float(video["fps"]) <= 0):
        raise ValueError("sample.video.fps must be positive and finite")
    for key in ("seed", "max_samples", "samples_per_input", "samples_per_prompt"):
        if key in sample and sample[key] is not None:
            minimum = 0 if key == "seed" else 1
            if type(sample[key]) is not int or sample[key] < minimum:
                raise ValueError(f"sample.{key} must be an integer >= {minimum}")
    return task


def sampling_config(task, config):
    """Independent effective config; generation settings never mutate loss settings."""
    result = deepcopy(config)
    sample = task.get("sample") or {}
    result["sampling"].update(sample.get("generation") or {})
    if sample.get("seed") is not None:
        result["sampling"]["seed"] = sample["seed"]
    result["data"]["video"].update(sample.get("video") or {})
    return result


def output_types(task, modality):
    validate_sampling_task(task)
    sample = task.get("sample") or {}
    outputs = sample.get("outputs", ["mp4"] if modality == "t2v" else ["text"])
    if isinstance(outputs, dict):
        outputs = outputs.get(modality, ["mp4"] if modality == "t2v" else ["text"])
    if not isinstance(outputs, list) or any(not isinstance(x, str) for x in outputs):
        raise ValueError("sample.outputs must be a list or a per-task mapping of lists")
    allowed = {"mp4", "latent"} if modality == "t2v" else {"text"}
    if set(outputs) - allowed:
        raise ValueError(f"Unsupported {modality} sample outputs: {set(outputs) - allowed}")
    outputs = set(outputs)
    collect = sample.get("collect") or {}
    if collect.get("latent_steps") or collect.get("trajectory") or collect.get("trace"):
        raise ValueError("Intermediate sampling trajectories are not supported; models are unchanged")
    if collect.get("final_latent") and modality == "t2v":
        outputs.add("latent")
    if modality == "t2v" and collect.get("initial_noise"):
        outputs.add("initial_noise")
    if modality == "v2t":
        outputs.update(key for key in ("token_ids", "text_logits") if collect.get(key))
    if not outputs:
        raise ValueError("Sampling requires at least one output artifact")
    return outputs


def build_requests(task, config, processor=None):
    """Plan ordinary sampling without opening RGB, ViT caches, or Wan latents."""
    validate_sampling_task(task)
    del processor  # Native tokenization is deferred until this worker owns a request.
    sample, data = task.get("sample") or {}, task.get("data") or {}
    mode = sample.get("mode") or task.get("benchmark")
    if mode not in {"t2v", "v2t", "mixed"}:
        raise ValueError("Ordinary sampling requires sample.mode=t2v/v2t/mixed")
    repetitions = sample.get("samples_per_input", sample.get("samples_per_prompt", 1))
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("samples_per_input must be a positive integer")
    base_seed = int(config["sampling"]["seed"] if sample.get("seed") is None else sample["seed"])
    rows = []
    prompts = data.get("prompts", [])
    if "prompt" in data:
        prompts = [data["prompt"], *prompts]
    for index, item in enumerate(prompts):
        if mode != "t2v":
            raise ValueError("Direct text prompts are only supported for T2V")
        prompt = item if isinstance(item, str) else item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("T2V prompts must be nonempty strings")
        rows.append({"task": "t2v", "sample_id": str(index), "prompt": prompt})
    sources = data.get("sources")
    if sources is None:
        sources = [] if prompts else config["data"].get("validation_sources", [])
    for source in sources:
        source = {"path": source} if isinstance(source, str) else deepcopy(source)
        if source.get("task") and mode != "mixed" and source["task"] != mode:
            continue
        # Mixed manifests retain their declared tasks; only simple caption-only
        # manifests inherit the task mode. Keep original indices after filtering.
        try:
            records = read_records(source, prompt_only=True)
        except ValueError as exc:
            if mode == "mixed" or "specify task=t2v/v2t" not in str(exc):
                raise
            source.setdefault("task", mode)
            records = read_records(source, prompt_only=True)
        for index, row in enumerate(records):
            if mode != "mixed" and row["task"] != mode:
                continue
            request = dict(task=row["task"], sample_id=row["sample_id"], source=source, index=index)
            if row["task"] == "t2v":
                request.update(prompt=_prompt(row["messages"]), messages=_target_layout_messages(row["messages"]))
            rows.append(request)
    maximum = sample.get("max_samples")
    if maximum is not None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError("sample.max_samples must be a positive integer")
        rows = rows[:maximum]
    if not rows:
        raise ValueError("Sampling requires data.prompts or data.sources")
    requests = []
    for index, row in enumerate(rows):
        output_types(task, row["task"])
        for repeat in range(repetitions):
            number = index * repetitions + repeat
            identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:10]
            requests.append(dict(row, id=f"{number:06d}-{_safe_id(row['sample_id'])}-{identity}",
                                 repetition=repeat, seed=base_seed + number,
                                 seed_policy=SEED_POLICY))
    return requests


def _artifact_path(output_dir, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Artifact path must stay inside its task directory: {relative}")
    root = Path(output_dir).resolve()
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Artifact path must stay inside its task directory: {relative}")
    return target


def artifact_paths(task, request):
    """Expected relative paths, shared by the dispatcher and artifact validation."""
    identifier = _safe_id(request["id"])
    defaults = {
        "mp4": request.get("path", f"videos/{identifier}.mp4"),
        "latent": f"latents/{identifier}.pt", "initial_noise": f"noise/{identifier}.pt",
        "text": f"texts/{identifier}.json", "token_ids": f"tokens/{identifier}.pt",
        "text_logits": f"logits/{identifier}.pt",
    }
    return {kind: defaults[kind] for kind in sorted(output_types(task, request["task"]))}


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the suffix so PyAV selects the same MP4 format as legacy saving.
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sample_request(runtime, task, request, output_dir):
    """Generate once, save requested final artifacts, return serializable metadata."""
    config = sampling_config(task, runtime.config)
    paths = artifact_paths(task, request)
    targets = {kind: _artifact_path(output_dir, path) for kind, path in paths.items()}
    model, processor = runtime.get_model(), runtime.processor
    settings, sampling = config["data"]["video"], config["sampling"]
    with evaluation_state(model), runtime.autocast():
        seed_evaluation(int(request["seed"]))
        if request["task"] == "t2v":
            batch = video_layout(request["prompt"], processor, config, messages=request.get("messages"))
        else:
            dataset = VideoDataset(request["source"], processor, config, prompt_only=True)
            batch = collate_video_samples(
                [dataset[request["index"]]], pad_token_id=processor.tokenizer.pad_token_id or 0,
            )
        batch = to_device(batch, runtime.device)
        values = {}
        if request["task"] == "t2v":
            shape = (16, (settings["num_frames"] - 1) // 4 + 1,
                     settings["height"] // 8, settings["width"] // 8)
            rng = torch.Generator(device=runtime.device).manual_seed(int(request["seed"]))
            kwargs = {}
            if "initial_noise" in paths:
                initial = torch.randn(shape, generator=rng, device=runtime.device, dtype=torch.float32)
                kwargs["initial_noise"] = [initial]
                values["initial_noise"] = {"noise": initial.cpu(), "seed": request["seed"]}
            latents = model.generate_video(
                batch, latent_shapes=[shape], steps=sampling["steps"], solver=sampling["solver"],
                cfg=sampling["cfg"], time_shift=sampling["time_shift"],
                timestep_scale=config["flow"]["timestep_scale"], rng=rng, **kwargs,
            )
            if "latent" in paths:
                values["latent"] = {"latent": latents[0].cpu(), "video": settings, "prompt": request["prompt"]}
            if "mp4" in paths:
                values["mp4"] = runtime.get_codec().decode(latents)[0]
        else:
            generated = model.generate_text(
                batch, max_new_tokens=sampling["max_new_tokens"], temperature=sampling["temperature"],
                top_k=sampling.get("top_k", 0), top_p=sampling.get("top_p", 1.0),
                eos_token_id=processor.tokenizer.eos_token_id, use_cache=False,
                return_logits="text_logits" in paths,
            )
            tokens = generated["sequences"] if isinstance(generated, dict) else generated
            if "text" in paths:
                values["text"] = {"text": processor.tokenizer.decode(tokens[0], skip_special_tokens=True)}
            if "token_ids" in paths:
                values["token_ids"] = {"token_ids": tokens[0].cpu()}
            if "text_logits" in paths:
                values["text_logits"] = {"logits": generated["logits"][0].cpu(), "token_ids": tokens[0].cpu()}
        for kind, path in targets.items():
            if kind == "mp4":
                _atomic_write(path, lambda p: write_video(p, values[kind], settings["fps"]))
            elif kind == "text":
                _atomic_write(path, lambda p: p.write_text(json.dumps(values[kind], ensure_ascii=False) + "\n"))
            else:
                _atomic_write(path, lambda p: torch.save(values[kind], p))
    return {"id": request["id"], "task": request["task"], "seed": request["seed"],
            "seed_policy": SEED_POLICY, "sample_id": request.get("sample_id", request["id"]),
            "artifacts": paths, "repetition": request.get("repetition", 0)}
