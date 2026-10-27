"""Verify outputs of tests/validation/test_mindspeed_resume.sh, including all state."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tests.validation.compare_mcore_checkpoints import compare
from utils.checkpoint_catalog import discover, latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    torch.set_num_threads(8)
    reports = {}
    pairs = dict(
        restore=(
            root / "before_resume_checkpoint",
            root / "restored/checkpoints/checkpoint-2",
        ),
        continuation=(
            root / "continuous/checkpoints/checkpoint-4",
            root / "interrupted/checkpoints/checkpoint-4",
        ),
    )
    for name, (left, right) in pairs.items():
        result = compare(left, right)
        (root / f"{name}-comparison.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        reports[name] = {
            key: value
            for key, value in result.items()
            if not key.endswith("differences")
        }
        reports[name]["tensor_difference_count"] = len(result["tensor_differences"])
        reports[name]["other_difference_count"] = len(result["other_differences"])

    retained = {}
    metrics = {}
    for experiment in ("continuous", "interrupted"):
        valid, rejected = discover(root / experiment / "checkpoints")
        retained[experiment] = [metadata["global_step"] for _, metadata in valid]
        assert retained[experiment] == [4, 3] and not rejected
        selected, _ = latest(root / experiment / "checkpoints")
        assert Path(selected).name == "checkpoint-4"
        rows = []
        for attempt in sorted((root / experiment / "logs/train").glob("attempt_*")):
            assert (
                json.loads((attempt / "launch_result.json").read_text())["exit_code"]
                == 0
            )
            assert (
                json.loads((attempt / "summary.json").read_text())["status"]
                == "completed"
            )
            config = json.loads((attempt / "run.json").read_text())["runtime"]
            assert config["backend"] == "MEGATRON" and config["world_size"] == 4
            assert config["distributed"]["overlap_grad_reduce"]
            assert config["distributed"]["overlap_param_gather"]
            gdn = config["qwen_gdn"]
            assert gdn["backend"] == "mindspeed"
            assert gdn["module"] == "mindspeed.core.ssm.npu_chunk_gated_delta_rule"
            assert gdn["cq_index_implementation"] == "batched_row_index_select_v1"
            assert gdn["triton_version"]
            for line in (attempt / "metrics.jsonl").read_text().splitlines():
                row = json.loads(line)
                rows.append(
                    {
                        key: row[key]
                        for key in (
                            "step",
                            "train/objective_global",
                            "train/loss_t2v_text",
                            "train/loss_t2v_video",
                            "train/loss_v2t_text",
                            "grad_norm",
                        )
                    }
                )
        metrics[experiment] = rows
        assert [row["step"] for row in rows] == [1, 2, 3, 4]

    before = json.loads((root / "logs-before-resume.json").read_text())
    assert before
    for name, digest in before.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert len(list((root / "interrupted/logs/train").glob("attempt_*"))) == 2
    assert len(list((root / "restore_logs/train").glob("attempt_*"))) == 1
    assert not (root / "restored/logs").exists()
    assert list(
        (root / "interrupted/checkpoints/quarantine").glob("checkpoint-3-*/partial")
    )
    reports.update(
        retained=retained,
        metrics=metrics,
        metrics_exact=metrics["continuous"] == metrics["interrupted"],
        previous_log_files_unchanged=len(before),
        separate_exp_and_override_logs=True,
        incomplete_checkpoint_quarantined=True,
    )
    (root / "verification_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)
    assert reports["restore"]["exact"], "Restoring a checkpoint changed its state"
    assert reports["continuation"]["exact"] and reports["metrics_exact"], (
        "Deterministic continuation differs; inspect the reports"
    )


if __name__ == "__main__":
    main()
