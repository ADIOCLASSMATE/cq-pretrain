"""Explicit scaling scenarios, not fabricated measurements of a 512-card run."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('summaries',nargs='+')
parser.add_argument('--target-cards',type=int,default=512)
parser.add_argument('--global-batch',type=int,help='Defaults to measured GBS times the card-count ratio')
parser.add_argument('--output',default='.runs/scaling.json')
args=parser.parse_args()
rows=[]
for file in args.summaries:
    measured=json.loads(Path(file).read_text())
    if measured.get('data_mode'):
        raise ValueError('Real-data epoch estimates require consumed documents/images; use analyze_moss_runs instead of capacity slots')
    if measured['profiling']:
        raise ValueError('Use unprofiled timing results for extrapolation')
    tp=measured['tp'];dp=args.target_cards//tp
    micro_batch=measured['micro_batch']
    global_batch=args.global_batch or measured['global_batch']*args.target_cards//measured['world']
    if args.target_cards%tp or global_batch%(dp*micro_batch):
        raise ValueError('TP and global batch must divide the proposed topology')
    if global_batch//(dp*micro_batch) != measured['grad_accum']:
        raise ValueError('Keep measured gradient accumulation for a comparable scaling extrapolation')
    basis=measured.get('wall_tokens_per_second',measured['tokens_per_second'])
    ideal=basis/measured['world']*args.target_cards
    rows.append(dict(source=file,measured=measured,target_cards=args.target_cards,tp=tp,dp=dp,
        global_batch=global_batch,micro_batch=micro_batch,
        grad_accum=global_batch//(dp*micro_batch),
        timing_basis='wall' if 'wall_tokens_per_second' in measured else 'training_step',
        scenarios=[dict(assumed_scaling_efficiency=e,tokens_per_second=ideal*e,
                       seconds_per_step=global_batch*measured['sequence_length']/(ideal*e))
                   for e in (1.,.8,.6)],
        caveat=f'Assumed {measured["world"]}-to-{args.target_cards} scaling efficiency, not a measurement or confidence interval. Larger DP collectives and stragglers remain unmeasured.'))
path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
