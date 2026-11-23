"""One benchmark task for one committed weight directory, on every local GPU.

Launched by scripts/evaluate.sh. Outputs are written in resumable pieces, so a
preempted job continues where it stopped when it is submitted again.
"""

import argparse
from pathlib import Path

import torch.distributed as dist

from evaluations.suites import TASKS, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("weights", type=Path, help="Evaluation snapshot or TP1 training checkpoint")
    parser.add_argument("output", type=Path, help="<output>/<task>/step_<step>/ receives the results")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke test on the first N items; writes SMOKE.json instead of RESULTS.json",
    )
    args = parser.parse_args()
    options = settings(args.config)
    from evaluations.suites.runtime import EvaluationModel

    runtime = EvaluationModel(args.weights, options["sampling"])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assets = options["assets"]
    if args.task in ("qa", "vqav2"):
        from evaluations.suites import qa

        qa.run(runtime, out, args.task, assets["benchmarks"], args.limit)
    elif args.task == "geneval":
        from evaluations.suites import geneval

        geneval.run(runtime, out, assets["geneval"], args.limit)
    else:
        from evaluations.suites import fid

        fid.run(runtime, out, assets["benchmarks"], args.limit)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
