"""Accept a distributed CUDA run only when every rank completed finite updates."""
import argparse
import json
import math
from pathlib import Path


def summarize(path, *, nodes=1, steps=1):
    root = Path(path)
    run = json.loads((root/'run.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    world = run['world_size']
    placements = run['rank_placement']
    assert {p['rank'] for p in placements} == set(range(world)), 'Missing rank placement'
    assert len({p['hostname'] for p in placements}) == nodes, 'Unexpected node count'
    assert run['measured_on'] == 'cuda'
    by_rank = {}
    for rank in range(world):
        rows = [json.loads(line) for line in (root/f'steps_rank_{rank:05d}.jsonl').read_text().splitlines()]
        # A retried attempt can legitimately repeat an uncheckpointed step.
        rows = {r['step']: r for r in rows}
        assert len(rows) >= steps, f'Rank {rank} completed too few steps'
        for row in rows.values():
            assert row['rank'] == rank and row['optimizer_updated'], f'Rank {rank} skipped an update'
            assert all(math.isfinite(row[k]) for k in ('loss','text_loss','image_loss','grad_norm','seconds'))
            assert row['seconds'] > 0 and row['peak_memory_gib'] > 0
        by_rank[rank] = rows
    completed = set(by_rank[0])
    assert all(set(rows) == completed for rows in by_rank.values()), 'Ranks completed different steps'
    assert max(completed) == summary['completed_updates'], 'Summary disagrees with rank receipts'
    tp = run['topology']['tp']
    placement = {p['rank']: p for p in placements}
    for first in range(0, world, tp):
        assert len({placement[r]['hostname'] for r in range(first, first+tp)}) == 1, 'TP crosses nodes'
    result = dict(passed=True, world=world, nodes=nodes, tp=tp, dp=run['topology']['dp'],
        completed_steps=sorted(completed), tokens_per_step=run['global_batch']*run['arguments']['sequence_length'],
        mean_seconds=summary['mean_seconds'], tokens_per_second=summary['tokens_per_second'],
        peak_memory_gib=max(r['peak_memory_gib'] for rows in by_rank.values() for r in rows.values()),
        per_rank=[dict(rank=rank, hostname=placement[rank]['hostname'],
            final_step=max(rows), final_loss=rows[max(rows)]['loss'],
            peak_memory_gib=max(r['peak_memory_gib'] for r in rows.values())) for rank, rows in by_rank.items()])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--nodes', type=int, default=1)
    p.add_argument('--steps', type=int, default=1)
    args = p.parse_args()
    result = summarize(args.run, nodes=args.nodes, steps=args.steps)
    (Path(args.run)/'all_ranks_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
