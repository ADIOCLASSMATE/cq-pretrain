"""Use Inspire's supported Jupyter transport when SSH bootstrap is unavailable.

Run with the Inspire CLI installation's Python. Uses its configured account;
never reads or prints credentials. Only targets the established Ascend notebook.
"""
import argparse
from pathlib import Path
from inspire.sdk import InspireClient

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--timeout', type=int, default=120)
args = parser.parse_args()
with InspireClient(account='wjx-ascend', operation_timeout=args.timeout + 60) as client:
    result = client.notebooks.exec('dev-wjx-ascend', workspace='昇腾卡公共空间',
        transport='jupyter', command=args.command, cwd=str(Path(__file__).resolve().parents[1]),
        timeout=args.timeout, max_output_bytes=16000)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    raise SystemExit(result.returncode)
