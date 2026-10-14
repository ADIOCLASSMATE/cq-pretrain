"""Optional direct comparison to the read-only origin/npu@193ec90 snapshot.

Run with UMM_SOURCE_SNAPSHOT=/path/to/snapshot. The installed package does not
import or require the old repository; these tests only certify a migration.
"""
import importlib
import os
from pathlib import Path

import pytest
import torch

import models
import utils
from data.dataset import ImageNetFlowCacheDataset
from data.packing import collate_imagenet_flow_cache, collate_segment_packed
from data.text_dataset import ClimbMixOnlineBatchDataset
from test_data import Tokenizer, image_files, text_files


@pytest.fixture
def source(monkeypatch):
    path = os.environ.get("UMM_SOURCE_SNAPSHOT")
    if not path:
        pytest.skip("set UMM_SOURCE_SNAPSHOT to compare with the pinned source tree")
    if not (Path(path) / "utils/dataset_imagenet_flow_cache.py").is_file():
        raise ValueError("UMM_SOURCE_SNAPSHOT is not the pinned unified-mm source")
    # The flat v3 tree and the source both have models/ and utils/. Only expose
    # the source's distinct helper modules during these optional comparisons.
    monkeypatch.setattr(models, "__path__", [*models.__path__, str(Path(path) / "models")])
    monkeypatch.setattr(utils, "__path__", [*utils.__path__, str(Path(path) / "utils")])
    return importlib.import_module("utils.dataset_imagenet_flow_cache")


def assert_common_equal(new, old):
    for key, expected in old.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(new[key], expected, rtol=0, atol=0)
        else:
            assert new[key] == expected, key


@pytest.mark.parametrize("task", ["class", "t2i", "i2t", "joint"])
def test_real_cache_sequence_and_posterior_are_identical_to_source(tmp_path, source, task):
    args = image_files(tmp_path)
    args.update(tokenizer=Tokenizer(), boi_token_id=3, eoi_token_id=4, mask_token_id=5,
                eos_token_id=2, seed=72, image_sigma_order="joint")
    if task == "class":
        args["conditioning_mode"] = "class"
    else:
        args["caption_sequence_modes"] = ["t2i", "i2t"] if task == "joint" else [task]
    new, old = ImageNetFlowCacheDataset(**args), source.ImageNetFlowCacheDataset(**args)
    for epoch in (0, 1, 13):
        new.set_epoch(epoch)
        old.set_epoch(epoch)
        for index in range(len(new)):
            assert_common_equal(new[index], old[index])


@pytest.mark.parametrize("modes", [["t2i"], ["t2i", "i2t"]])
def test_flat_and_packed_batches_match_source(tmp_path, source, modes):
    args = dict(**image_files(tmp_path), tokenizer=Tokenizer(), boi_token_id=3, eoi_token_id=4,
                mask_token_id=5, eos_token_id=2, image_sigma_order="joint", caption_sequence_modes=modes)
    new, old = ImageNetFlowCacheDataset(**args), source.ImageNetFlowCacheDataset(**args)
    items_new, items_old = [new[i] for i in range(4)], [old[i] for i in range(4)]
    old_pack = importlib.import_module("utils.multimodal_segment_packing")
    assert_common_equal(collate_imagenet_flow_cache(items_new), source.collate_imagenet_flow_cache(items_old))
    options = dict(nominal_capacity=64, image_uncond_prob=.4, emit_audit_manifest=True)
    assert_common_equal(collate_segment_packed(items_new, **options), old_pack.collate_segment_packed(items_old, **options))


def test_online_text_stream_and_resume_match_source(tmp_path, source):
    old_text = importlib.import_module("utils.climbmix_online_dataset")
    options = dict(shard_paths=text_files(tmp_path), tokenizer=Tokenizer(), eos_token_id=2,
                   sequence_length=8, micro_batch_size=2, tokenizer_batch_documents=2, seed=43)
    new = iter(ClimbMixOnlineBatchDataset(**options))
    old = iter(old_text.ClimbMixOnlineBatchDataset(**options))
    for _ in range(7):
        new_batch, old_batch = next(new), next(old)
        assert_common_equal(new_batch, old_batch)
    resumed = iter(ClimbMixOnlineBatchDataset(**options, resume_state=new_batch["stream_state"]))
    assert_common_equal(next(resumed), next(old))
