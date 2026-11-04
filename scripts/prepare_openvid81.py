#!/usr/bin/env python3
"""Run four independent resumable workers, then verify both complete OpenVid indices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from data.manifest import atomic_write, file_sha256, original_digest, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/prepare_openvid81.yaml")
    parser.add_argument("--devices", default="0,1,2,3")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    devices = [int(x) for x in args.devices.split(",")]
    if len(devices) != len(set(devices)) or any(x < 0 for x in devices):
        raise ValueError("Devices must be unique nonnegative indices")
    resolutions = ("832x480", "960x544")
    folder = root / "metadata" / "latent_validation"
    # The large run is deliberately gated on actual extraction, online parity and resume.
    for resolution in resolutions:
        for phase in ("pilot", "pilot_resume"):
            result = json.loads(
                (
                    root
                    / resolution
                    / "metadata"
                    / "latent_runs"
                    / f"{phase}.shard-00.json"
                ).read_text()
            )
            if not result["complete"] or result["completed"] != 9:
                raise ValueError(f"{resolution}: {phase} has not passed")
            if phase == "pilot" and result["verified_online"]["max_abs_error"] != {
                "qwen": 0.0,
                "wan": 0.0,
            }:
                raise ValueError("Online parity pilot has not passed")
            if phase == "pilot_resume" and any(result["encoded"].values()):
                raise ValueError("Resume unexpectedly encoded data")
    before = json.loads((folder / "pilot_files_before_resume.json").read_text())
    for name, stat in before.items():
        current = Path(name).stat()
        if current.st_mtime_ns != stat["mtime_ns"] or current.st_size != stat["bytes"]:
            raise ValueError(f"Resume changed an existing latent: {name}")
    status = {
        "phase": "extracting",
        "started": time.time(),
        "pid": os.getpid(),
        "devices": devices,
        "complete": False,
        "failures": [],
    }
    status_path = folder / "full_status.json"

    def write_status():
        status["elapsed_seconds"] = time.time() - status["started"]
        status["workers"] = {}
        for resolution in resolutions:
            for shard in range(len(devices)):
                report = (
                    root
                    / resolution
                    / "metadata"
                    / "latent_runs"
                    / f"full.shard-{shard:02d}.json"
                )
                if report.exists():
                    status["workers"][f"{resolution}/{shard}"] = json.loads(
                        report.read_text()
                    )
        atomic_write(
            status_path, json.dumps(status, ensure_ascii=False, indent=2) + "\n"
        )

    def run(resolution, shard, device, verify=False):
        phase = "verify_full" if verify else "full"
        command = [
            sys.executable,
            "-m",
            "scripts.prepare_latents",
            "--config",
            args.config,
            "--manifest",
            str(root / resolution / "samples.jsonl"),
            "--in-place",
            "--run-id",
            phase,
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
        )
        if verify:
            command += [
                "--verify-only",
                "--set",
                "training.device=cpu",
                "--set",
                "distributed.backend=accelerate",
            ]
            env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        else:
            command += [
                "--num-shards",
                str(len(devices)),
                "--shard-index",
                str(shard),
                "--device-index",
                str(device),
            ]
        log = folder / f"{phase}_{resolution}_shard-{shard:02d}.log"
        with log.open("a") as handle:
            handle.write(
                json.dumps({"started": time.time(), "command": command}) + "\n"
            )
            handle.flush()
            process = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env
            )
            code = process.wait()
        if code:
            raise RuntimeError(
                f"{phase} {resolution} shard {shard} exited {code}; see {log}"
            )

    def worker(shard, device):
        for resolution in resolutions:
            run(resolution, shard, device)

    def wait_for_jobs(futures):
        while not all(future.done() for future in futures):
            write_status()
            time.sleep(15)
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                status["failures"].append(str(exc))
        write_status()
        if status["failures"]:
            raise RuntimeError("; ".join(status["failures"]))

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            wait_for_jobs(
                [
                    pool.submit(worker, shard, device)
                    for shard, device in enumerate(devices)
                ]
            )
        status["phase"] = "verifying"
        with ThreadPoolExecutor(max_workers=2) as pool:
            wait_for_jobs(
                [pool.submit(run, resolution, 0, 0, True) for resolution in resolutions]
            )
        summary, paired = {}, []
        for resolution in resolutions:
            manifest = root / resolution / "samples.jsonl"
            rows = read_manifest(manifest)
            if len(rows) != 3020:
                raise ValueError(f"{resolution}: expected 3020 rows")
            baseline = json.loads(
                (
                    manifest.parent / "metadata/latent_runs/original_fields.json"
                ).read_text()
            )["samples"]
            if list(baseline) != [row["sample_id"] for row in rows]:
                raise ValueError("Sample order changed")
            for row in rows:
                if original_digest(row) != baseline[row["sample_id"]]:
                    raise ValueError("Original sample fields changed")
                if not all(row.get(f"{kind}_cache") for kind in ("qwen", "wan")):
                    raise ValueError("Incomplete latent references")
            paired.append(
                [
                    (r["sample_id"], r["caption"], r["fps_num"], r["fps_den"])
                    for r in rows
                ]
            )
            report = json.loads(
                (
                    manifest.parent / "metadata/latent_runs/verify_full.shard-00.json"
                ).read_text()
            )
            if (
                not report["complete"]
                or report["completed"] != 3020
                or any(report["encoded"].values())
            ):
                raise ValueError("Full verification failed")
            summary[resolution] = {
                "rows": len(rows),
                "index_sha256": file_sha256(manifest),
                "bytes": sum(
                    r["latent_info"][k]["file_bytes"]
                    for r in rows
                    for k in ("qwen", "wan")
                ),
                "dtypes": {
                    k: dict(Counter(r["latent_info"][k]["dtype"] for r in rows))
                    for k in ("qwen", "wan")
                },
                "shapes": {
                    k: dict(Counter(str(r["latent_info"][k]["shape"]) for r in rows))
                    for k in ("qwen", "wan")
                },
            }
        if paired[0] != paired[1]:
            raise ValueError("Resolution indices are not aligned")
        status.update(phase="complete", complete=True, summary=summary)
    except BaseException as exc:
        status.update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_status()
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
