"""Audit identical consumed batches and BF16 training metrics across optimizer groups."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(root):
    root=Path(root);results={}
    for method in ('cq','bi'):
        baseline=root/f'{method}_oi1'
        reference=[json.loads(x) for x in (baseline/'steps.jsonl').read_text().splitlines()]
        traces={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.glob('consumed_rank_*.jsonl')}
        assert len(traces)==512
        cases=[]
        for instances in (1,4,16):
            path=root/f'{method}_oi{instances}'
            rows=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
            assert len(rows)==len(reference)==16
            actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('consumed_rank_*.jsonl')}
            assert traces==actual,f'Consumed batch identity differs: {path.name}'
            errors={k:0. for k in ('loss','text_loss','image_loss','grad_norm')}
            violations=[]
            for a,b in zip(reference,rows):
                for key in ('step','body_tokens','occupied_tokens','images_consumed','active_microbatches'):
                    assert a[key]==b[key],(path.name,key)
                for key in errors:
                    error=abs(a[key]-b[key]);errors[key]=max(errors[key],error)
                    if error>1e-5+3e-4*abs(a[key]):
                        violations.append(dict(step=a['step'],metric=key,baseline=a[key],candidate=b[key],
                            absolute_error=error,relative_error=error/max(abs(a[key]),1e-30)))
            summary=json.loads((path/'summary.json').read_text())
            cases.append(dict(instances=instances,wall_tokens_per_second=summary['wall_tokens_per_second'],
                peak_memory_gib=summary['peak_memory_gib'],mean_seconds=summary['mean_seconds'],
                max_abs_metric_errors=errors,consumed_traces_bitwise_identical=True,
                metrics_within_tolerance=not violations,violations=violations))
        results[method]=cases
    report=dict(passed=all(c['metrics_within_tolerance'] for cases in results.values() for c in cases),
        default_optimizer_instances=1,world=512,steps=16,metric_atol=1e-5,metric_rtol=3e-4,
        scope='Consumed batch traces are bitwise; loss/norm tolerance is not a full optimizer tensor equivalence test',results=results)
    (root/'comparison.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();result=compare(a.root)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
