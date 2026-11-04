"""Test atomic mkdir claims with 16 simultaneous contenders on each of two hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import time
from data.preprocess.manifest import atomic_json
from data.preprocess.parallel_feasibility.probe import PROBE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--role", choices=["local", "remote"], required=True)
parser.add_argument("--start", type=float, required=True)
args = parser.parse_args()
root = PROBE / "claim_race"
root.mkdir(exist_ok=True)
if time.time() >= args.start:
    raise RuntimeError("Both hosts must start before the agreed race time")


def compete(worker):
    won = []
    for round_id in range(50):
        target_time = args.start+round_id*0.05
        time.sleep(max(0, target_time-time.time()))
        claim = root / f"round-{round_id:02d}"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        won.append(round_id)
        atomic_json(claim / "owner.json", dict(host=platform.node(), role=args.role, worker=worker))
    return won


with ThreadPoolExecutor(max_workers=16) as pool:
    wins = [round_id for winners in pool.map(compete, range(16)) for round_id in winners]
report = dict(host=platform.node(), role=args.role, contenders=16, rounds=50,
              start_epoch=args.start, wins=wins, wins_count=len(wins), completed=True)
atomic_json(PROBE / f"claim_race_{args.role}.json", report)
print(json.dumps(report))
