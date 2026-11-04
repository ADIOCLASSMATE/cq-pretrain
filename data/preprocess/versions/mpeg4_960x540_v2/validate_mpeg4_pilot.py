"""Check the production pilot against the previously measured MPEG-4 q=2 files."""
import argparse
import csv
import json
from pathlib import Path

from .manifest import atomic_json, read_jsonl, sha256
from .openvidhd import ANALYSIS, load_config, now


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    config = load_config(root)
    require(config["pilot"] and config["candidates"] == 10, "Expected ten-sample pilot")
    audit = json.loads((root / "validation.json").read_text())
    require(audit["all_passed"] and audit["source_correspondence"] and audit["verified"] == 10,
            "Independent source audit must pass first")
    require(audit["config_fingerprint"] == config["fingerprint"], "Audit/config mismatch")
    comparison = ANALYSIS / "codec_comparison_10"
    comparison_config = json.loads((comparison / "comparison_config.json").read_text())
    originals = {r["sample_id"]: r for r in read_jsonl(Path(comparison_config["pilot"]) / "samples.jsonl")}
    with (comparison / "per_sample.csv").open() as handle:
        measured = {r["sample_id"]: r for r in csv.DictReader(handle) if r["mode"] == "mpeg4_q2"}
    records = read_jsonl(root / "samples.jsonl")
    require(len(records) == 10 and {r["sample_id"] for r in records} == set(measured) == set(originals),
            "Pilot sample identities differ from codec comparison")
    audits = {r["sample_id"]: r for r in audit["samples"]}
    for record in records:
        name = record["sample_id"]
        expected = measured[name]
        original = originals[name]
        require(record["caption"] == original["caption"] and record["source"] == original["source"],
                f"Original caption/source changed: {name}")
        require(record["validation"]["pre_encoder_rgb_sha256"] == original["validation"]["rgb_sha256"],
                f"Resized stride-2 RGB input changed: {name}")
        for key in ("num_frames", "fps_num", "fps_den", "output_bytes"):
            require(record[key] == int(expected[key]), f"Changed {key}: {name}")
        # Same input, runtime and single-threaded native MPEG-4 settings should
        # reproduce the already reviewed bitstream, including its exact PTS.
        require(sha256(root / record["video_path"]) == sha256(comparison / expected["video_path"])
                == record["output_sha256"] == expected["sha256"], f"Benchmark MP4 differs: {name}")
        for actual_key, measured_key in (("rgb_sse", "sse"), ("rgb_sae", "sae"), ("rgb_values", "values")):
            require(audits[name][actual_key] == int(expected[measured_key]), f"Quality error changed: {name}")
    report = dict(updated_at=now(), all_passed=True, verified=10,
                  output_frames=sum(r["num_frames"] for r in records),
                  output_bytes=sum(r["output_bytes"] for r in records),
                  original_ids_captions_and_source_metadata_unchanged=True,
                  pre_encoder_rgb_matches_lossless_pilot=True,
                  mp4_bytes_match_reviewed_mpeg4_q2=True,
                  rgb_error_statistics_match_comparison=True,
                  config_fingerprint=config["fingerprint"], comparison_root=str(comparison))
    atomic_json(root / "benchmark_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
