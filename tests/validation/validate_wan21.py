"""Audit a real local Wan checkpoint and exercise its full generator on a device.

From the repository root, with the project and device runtime installed::

    python -m tests.validation.validate_wan21 --checkpoint /path/to/Wan2.1-T2V-1.3B \
        --device npu:0 --report outputs/wan21-validation/report.json

CUDA is also supported. Uses synthetic noisy VAE latents and backbone conditions;
does not load a VAE/backbone, update weights, download assets or assess video quality.
All native weights are compared exactly after BF16 conversion, followed by full
model forward/backward checks. A supplied download_manifest.json is hash-checked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.generator_wan21 import Wan21Generator


def save_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def audit_checkpoint(model, directory):
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
              else ["diffusion_pytorch_model.safetensors"])
    manifest_path = directory / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {row["path"]: row for row in manifest.get("files", [])}
    state = model.state_dict()
    seen, files, parameter_count = set(), [], 0
    for name in shards:
        path = directory / name
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = entries.get(name, {}).get("sha256")
        if expected is not None and digest != expected:
            raise AssertionError(f"Checkpoint checksum mismatch: {name}")
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": digest,
                      "manifest_hash_matched": expected == digest if expected else None})
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                assert key not in seen, f"Duplicate checkpoint key: {key}"
                source = handle.get_tensor(key)
                assert key in state and torch.equal(state[key], source.to(state[key].dtype)), key
                seen.add(key)
                parameter_count += source.numel()
    assert set(state) - seen == {"condition_proj.weight", "condition_proj.bias"}
    assert all(not p.is_meta for p in model.parameters())
    return {"repo": manifest.get("repo"), "revision": manifest.get("revision"), "files": files,
            "native_tensors_exact_after_bfloat16_conversion": len(seen),
            "native_parameter_count": parameter_count,
            "new_parameters": ["condition_proj.weight", "condition_proj.bias"]}


def compare(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    return {"max_abs_error": (actual - expected).abs().max().item(),
            "relative_l2_error": ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()}


def gradient_norm(value):
    assert value is not None and bool(torch.isfinite(value).all())
    norm = value.detach().float().norm().item()
    assert norm > 0, "Gradient unexpectedly zero"
    return norm


def check_forward(model, device):
    shapes = [(model.in_dim, 3, 8, 12), (model.in_dim, 5, 6, 8)]
    latents = [torch.randn(shape, device=device) for shape in shapes]
    times = torch.tensor([150., 850.], device=device)
    condition = torch.randn(2, 11, model.condition_dim, device=device)
    mask = torch.tensor([[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        batched = model(latents, times, condition, condition_mask=mask)
        parity = []
        for i, actual in enumerate(batched):
            assert actual.shape == latents[i].shape and actual.dtype == torch.float32
            expected, = model([latents[i]], times[i:i + 1], condition[i, mask[i]][None])
            parity.append(compare(actual, expected))
        poisoned = condition.masked_fill(~mask[..., None], float("nan"))
        masked = model(latents, times, poisoned, condition_mask=mask)
        for actual, expected in zip(masked, batched):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        changed = condition.clone()
        changed[1] += 3
        separate = model(latents, times, changed, condition_mask=mask)
        torch.testing.assert_close(separate[0], batched[0], atol=0, rtol=0)
        assert not torch.allclose(separate[1], batched[1])
        long_condition = torch.randn(1, 519, model.condition_dim, device=device)
        long_output, = model([latents[0]], times[:1], long_condition)
        assert long_output.shape == latents[0].shape and bool(torch.isfinite(long_output).all())
    return (latents, times, poisoned, mask), {
        "latent_shapes_CTHW": shapes, "native_timesteps": times.cpu().tolist(),
        "valid_condition_lengths": mask.sum(-1).cpu().tolist(),
        "batched_vs_individual": parity, "masked_nan_invariance_exact": True,
        "sample_isolation_exact": True, "valid_condition_changes_prediction": True,
        "condition_length_519_passed": True,
    }


def check_backward(model, inputs):
    latents, times, condition, mask = inputs
    latents = [x.detach().requires_grad_() for x in latents]
    condition = condition.detach().requires_grad_()
    model.train()
    model.enable_gradient_checkpointing()
    predictions = model(latents, times, condition, condition_mask=mask)
    # Independent per-video means avoid giving longer videos more loss weight.
    targets = [torch.randn_like(x) for x in predictions]
    loss = sum((x - target).square().mean() for x, target in zip(predictions, targets)) / len(targets)
    assert bool(torch.isfinite(loss))
    loss.backward()
    latent_norms = [gradient_norm(x.grad) for x in latents]
    condition_norm = gradient_norm(condition.grad[mask])
    assert torch.count_nonzero(condition.grad[~mask]).item() == 0
    parameters = dict(model.named_parameters())
    checks = []
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        checks.append(torch.isfinite(parameter.grad).all())
    assert bool(torch.stack(checks).all()), "Nonfinite parameter gradient"
    names = ("condition_proj.weight", "patch_embedding.weight", "text_embedding.0.weight",
             "time_embedding.0.weight", "time_projection.1.weight", "blocks.0.self_attn.q.weight",
             "blocks.0.cross_attn.k.weight", f"blocks.{model.num_layers - 1}.self_attn.q.weight",
             "head.head.weight")
    return {"loss": loss.detach().item(), "gradient_checkpointing": model.gradient_checkpointing,
            "latent_gradient_norms": latent_norms, "valid_condition_gradient_norm": condition_norm,
            "masked_condition_gradients_exactly_zero": True,
            "parameter_tensors_with_finite_gradients": len(checks),
            "selected_parameter_gradient_norms": {name: gradient_norm(parameters[name].grad) for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True, help="An actual accelerator, e.g. npu:0 or cuda:0")
    parser.add_argument("--condition-dim", type=int, default=1024)
    parser.add_argument("--report", type=Path, default=Path("output/wan21-validation/report.json"))
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
              "condition_dim": args.condition_dim, "dtype": "bfloat16", "seed": 84,
              "torch": torch.__version__}
    try:
        if args.device.startswith("npu"):
            import torch_npu
            report["torch_npu"] = torch_npu.__version__
        import diffusers
        report["diffusers"] = diffusers.__version__
        device = torch.device(args.device)
        assert device.type in {"npu", "cuda"}, "An actual accelerator is required"
        runtime = getattr(torch, device.type)
        runtime.set_device(device)
        report["device_name"] = runtime.get_device_name(device)
        torch.set_num_threads(8)
        torch.manual_seed(84)
        print("Loading the complete native Wan checkpoint on CPU ...", flush=True)
        model = Wan21Generator.from_wan_pretrained(
            args.checkpoint, condition_dim=args.condition_dim, torch_dtype=torch.bfloat16,
            attention_backend="auto", gradient_checkpointing=True,
        )
        report["checkpoint_audit"] = audit_checkpoint(model, args.checkpoint)
        save_report(report, args.report)
        print("All native checkpoint tensors matched; moving full model to accelerator ...", flush=True)
        model.to(device)
        runtime.reset_peak_memory_stats(device)
        inputs, report["forward"] = check_forward(model, device)
        save_report(report, args.report)
        print("Variable latents, condition masks and independent-call parity passed; backward ...", flush=True)
        report["backward"] = check_backward(model, inputs)
        runtime.synchronize(device)
        report["peak_device_memory_allocated_bytes"] = runtime.max_memory_allocated(device)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_report(report, args.report)
        print(f"Validation {report['status']}: {args.report}", flush=True)


if __name__ == "__main__":
    main()
