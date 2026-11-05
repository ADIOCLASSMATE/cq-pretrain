"""Communication scheduling and loss weighting independent of NPU availability."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from training.loss import global_loss_weights
from training.runtime import MCoreRuntime, _gdn_contract
from distributed.communication import OrderedGradientBuckets, OrderedParameterGathers
from tests.test_generator_wan21 import generator


def test_runtime_audits_gdn_provider_but_allows_identical_checkout_relocation():
    identity = dict(backend="mindspeed", provider="MindSpeed", source_sha256="abc",
                    file="/old/kernel.py", triton_version="3.2.1",
                    cq_index_implementation="batched_row_index_select_v1")
    runtime = MCoreRuntime.__new__(MCoreRuntime)
    runtime.gdn_identity = identity
    actual = dict(identity, file="/new/kernel.py")
    model = SimpleNamespace(backbone=SimpleNamespace(gdn_backend_identity=lambda: actual))
    runtime._audit_gdn_backend(model)
    runtime._audit_gdn_backend(torch.nn.Linear(2, 1))
    runtime._audit_gdn_backend(SimpleNamespace(backbone=torch.nn.ModuleDict()))
    with pytest.raises(RuntimeError, match="does not expose"):
        runtime._audit_gdn_backend(SimpleNamespace(
            backbone=torch.nn.ModuleDict(), config=SimpleNamespace(qwen_config={})
        ))
    assert "file" not in _gdn_contract(identity)
    for key, replacement in (("backend", "native"), ("source_sha256", "def"),
                             ("cq_index_implementation", "old_gather")):
        actual[key] = replacement
        with pytest.raises(RuntimeError, match="runtime contract"):
            runtime._audit_gdn_backend(model)
        actual[key] = identity[key]


@pytest.mark.parametrize("source", ["scratch", "native_components", "export"])
def test_all_loading_paths_persist_gdn_backend_in_both_config_copies(source, tmp_path, monkeypatch):
    from configs.config import DEFAULTS
    from models.modeling_umm import UMM
    from tests.test_modeling_umm_video import configuration
    from training.train import load_model

    architecture = configuration(layers=4, freeze=True)
    config = deepcopy(DEFAULTS)
    config["model"].update(architecture=architecture.to_dict(),
                           mask_token_id=59, video_mask_token_id=60,
                           gradient_checkpointing=False, qwen_gdn_backend="mindspeed")
    if source == "scratch":
        config["model"]["from_scratch"] = True
    elif source == "native_components":
        monkeypatch.setattr(UMM, "from_pretrained_components", lambda *args, **kwargs: UMM(architecture))
    else:
        export = tmp_path / "weights.pt"
        torch.save(dict(schema="umm_weights_v2", model=UMM(architecture).state_dict(),
                        config=deepcopy(config)), export)
        config["model"]["checkpoint"] = str(export)
    model = load_model(config)
    assert model.backbone.config.qwen_gdn_backend == "mindspeed"
    assert model.config.qwen_config.text_config.qwen_gdn_backend == "mindspeed"
    assert model.backbone.config is not model.config.qwen_config.text_config
    saved = config["model"]["architecture"]["qwen_config"]["text_config"]
    assert saved["qwen_gdn_backend"] == "mindspeed"
    restored = UMM(type(model.config).from_dict(config["model"]["architecture"]))
    assert restored.backbone.config.qwen_gdn_backend == "mindspeed"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_time_module_hooks_preserve_fp32_values_and_gradients(dtype):
    model = generator().to(dtype)
    reference = deepcopy(model)
    calls = []
    for name in ("time_embedding.0", "time_embedding.2", "time_projection.1"):
        model.get_submodule(name).register_forward_pre_hook(
            lambda module, args, name=name: calls.append(name)
        )
    times = torch.tensor([12.0, 501.0], requires_grad=True)
    old_times = times.detach().clone().requires_grad_()
    half = model.freq_dim // 2
    phase = times[:, None] * 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)
    hidden = model.time_embedding(torch.cat((phase.cos(), phase.sin()), -1))
    modulation = model.time_projection(hidden).unflatten(1, (6, model.dim))
    half = reference.freq_dim // 2
    frequencies = torch.pow(10000.0, -torch.arange(half, dtype=torch.float32) / half)
    angles = old_times[:, None] * frequencies[None]
    expected = torch.cat((angles.cos(), angles.sin()), -1)
    for layer in reference.time_embedding:
        expected = (
            torch.nn.functional.linear(
                expected, layer.weight.float(), layer.bias.float()
            )
            if isinstance(layer, torch.nn.Linear)
            else layer(expected)
        )
    projection = reference.time_projection[1]
    expected_modulation = torch.nn.functional.linear(
        reference.time_projection[0](expected),
        projection.weight.float(),
        projection.bias.float(),
    ).unflatten(1, (6, model.dim))
    assert calls == ["time_embedding.0", "time_embedding.2", "time_projection.1"]
    assert hidden.dtype == modulation.dtype == torch.float32
    torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
    torch.testing.assert_close(modulation, expected_modulation, rtol=0, atol=0)
    (hidden.square().sum() + modulation.square().sum()).backward()
    (expected.square().sum() + expected_modulation.square().sum()).backward()
    torch.testing.assert_close(times.grad, old_times.grad, rtol=0, atol=0)
    for (name, actual), (_, original) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        if actual.grad is not None:
            torch.testing.assert_close(
                actual.grad, original.grad, rtol=0, atol=0, msg=name
            )


def test_global_counts_weight_tokens_not_microbatch_means():
    # Rank 0 has 2+8 text tokens, rank 1 has 6+4, three videos globally.
    batches = [
        dict(
            labels=torch.zeros(1, n, dtype=torch.long),
            token_types=torch.zeros(1, n, dtype=torch.long),
            target_video_ids=list(range(v)),
        )
        for n, v in ((2, 1), (8, 0))
    ]
    runtime = SimpleNamespace(
        device="cpu",
        num_processes=2,
        reduce=lambda tensor, **kw: tensor + torch.tensor([10, 2]),
    )
    weights = global_loss_weights(batches, runtime)
    assert weights == [(0.2, 2 / 3), (0.8, 0.0)]
    x = torch.tensor(3.0, requires_grad=True)
    sum(x * w[0] for w in weights).backward()
    assert x.grad == 1.0  # Not divided again by the two accumulation steps.


class Bucket:
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.ddp_config = SimpleNamespace(
            overlap_param_gather=True, overlap_grad_reduce=True
        )
        self.params = [torch.nn.Parameter(torch.ones(2))]
        self.params[0].main_grad = torch.full((2,), 7.0)
        self.per_param_grad_ready_counts = {}
        self.param_gather_dispatched = False

    def start_grad_sync(self, force_all_reduce=False):
        self.events.append(("grad", self.index))

    def start_param_sync(self, force_sync=False):
        self.events.append(("param", self.index))
        self.param_gather_dispatched = True

    def register_grad_ready(self, p, force):
        self.per_param_grad_ready_counts[p] = 1
        self.start_grad_sync(force)


def test_missing_final_branch_preserves_accumulated_gradients_and_order():
    events = []
    groups = [Bucket(i, events) for i in range(3)]
    wrapped = SimpleNamespace(
        bucket_groups=groups, expert_parallel_bucket_groups=[], force_all_reduce=False
    )
    ordered = OrderedGradientBuckets(wrapped)
    groups[2].register_grad_ready(groups[2].params[0], False)
    assert not events
    assert ordered.finish_backward() == 2
    assert events == [("grad", 0), ("grad", 1), ("grad", 2)]
    for group in groups:
        assert torch.equal(group.params[0].main_grad, torch.full((2,), 7.0))


def test_skipped_optional_module_still_gathers_in_fixed_order():
    events = []
    groups = [Bucket(i, events) for i in range(3)]
    wrapped = SimpleNamespace(bucket_groups=groups, expert_parallel_bucket_groups=[])
    OrderedParameterGathers(wrapped)
    groups[0].start_param_sync()
    assert events == [("param", 2), ("param", 1), ("param", 0)]
    groups[1].start_param_sync()
    assert len(events) == 3
