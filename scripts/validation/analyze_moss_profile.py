"""Summarize exported Ascend profiles without treating profiler wall time as throughput."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def summarize_kernel_pmu(path):
    """Duration-weighted per-kernel counters, never whole-device utilization."""
    if not path.exists():return []
    metrics=('aic_mac_ratio','aic_mte2_ratio','aiv_vec_ratio','aiv_mte3_ratio','cube_utilization(%)')
    totals=defaultdict(lambda:defaultdict(float))
    with path.open(newline='') as source:
        for row in csv.DictReader(source):
            duration=float(row['Duration(us)'])
            values=totals[row['Type']]
            values['duration_us']+=duration
            values['count']+=1
            for key in metrics:
                try:value=float(row[key])
                except (KeyError,ValueError,TypeError):continue
                if not math.isfinite(value):continue
                values[key]+=value*duration
                values[key+'_weight']+=duration
    return [dict(op=op,count=int(values['count']),summed_kernel_seconds=values['duration_us']/1e6,
                 duration_weighted_metrics={key:values[key]/values[key+'_weight'] for key in metrics
                                            if values[key+'_weight']>0})
            for op,values in sorted(totals.items(),key=lambda item:item[1]['duration_us'],reverse=True)
            if any(values[key+'_weight']>0 for key in metrics)]


def summarize(path):
    traces=read_csv(path/'step_trace_time.csv')
    steps=[]
    for row in traces:
        stage=float(row['Stage'])
        if stage<=0:continue
        values={key:float(row[key])/1e6 for key in
                ('Computing','Communication(Not Overlapped)','Communication','Overlapped','Free','Stage')}
        steps.append(dict(step=row['Step'],seconds=values,
            compute_fraction=values['Computing']/values['Stage'],
            nonoverlapped_communication_fraction=values['Communication(Not Overlapped)']/values['Stage'],
            free_fraction=values['Free']/values['Stage']))
    ops=defaultdict(lambda:dict(count=0,seconds=0.))
    core_seconds=defaultdict(float)
    op_core_types=defaultdict(set)
    for row in read_csv(path/'op_statistic.csv'):
        value=ops[row['OP Type']]
        seconds=float(row['Total Time(us)'])/1e6
        value['count']+=int(row['Count']);value['seconds']+=seconds
        core=row.get('Core Type','unknown')
        core_seconds[core]+=seconds;op_core_types[row['OP Type']].add(core)
    kernel_seconds=sum(v['seconds'] for v in ops.values())
    ranked=[dict(op=name,**values,core_types=sorted(op_core_types[name]),
                 summed_kernel_duration_fraction=values['seconds']/kernel_seconds)
            for name,values in sorted(ops.items(),key=lambda x:x[1]['seconds'],reverse=True)]
    communication=[]
    source=path/'communication.json'
    if source.exists():
        for step,groups in json.loads(source.read_text()).items():
            kinds=defaultdict(lambda:dict(count=0,elapsed_ms=0.,transit_ms=0.,wait_ms=0.,synchronization_ms=0.,idle_ms=0.))
            for group in groups.values():
                for name,info in group.items():
                    if not name.startswith('hcom_'):continue
                    time_info=info.get('Communication Time Info',{})
                    if not time_info:continue
                    kind=name.split('__')[0].removeprefix('hcom_')
                    value=kinds[kind];value['count']+=1
                    for output,input_key in (('elapsed_ms','Elapse Time(ms)'),('transit_ms','Transit Time(ms)'),
                            ('wait_ms','Wait Time(ms)'),('synchronization_ms','Synchronization Time(ms)'),('idle_ms','Idle Time(ms)')):
                        value[output]+=time_info.get(input_key,0.)
            communication.append(dict(step=step,collectives=dict(kinds)))
    return dict(source=str(path),steps=steps,
        mean_step_seconds={key:statistics.mean(row['seconds'][key] for row in steps) for key in steps[0]['seconds']} if steps else {},
        summed_kernel_seconds=kernel_seconds,summed_kernel_seconds_by_core_type=dict(core_seconds),
        operators=ranked,communication=communication,
        kernel_pmu=summarize_kernel_pmu(path/'kernel_details.csv'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path,help='Stage profile directory containing rank subdirectories')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(args.profile.glob('rank*/*/ASCEND_PROFILER_OUTPUT'))
    paths=[p for p in paths if (p/'step_trace_time.csv').exists() and (p/'op_statistic.csv').exists()]
    if not paths:raise ValueError('No completed exported profiles found')
    result=dict(caveats=[
        'Profiled steps identify bottlenecks; profiler wall time is excluded from epoch throughput.',
        'Kernel duration sums may overlap and are not fractions of end-to-end wall time.',
        'MC2 fused matmul-allreduce and its AI_CPU controller include communication and synchronization. Profiler Computing is not pure matrix compute, and controller duration is not an additional serial wall-time cost.',
        'Communication elapsed time includes rank waiting; it is not exclusively network transfer.',
        'Kernel PMU ratios describe measured pipeline activity, not whole-step FLOP or bandwidth utilization.',
        'HCCS transfers may appear under Idle in Communication Time Info; use link details before labeling idle as rank waiting.'],
        ranks=[summarize(path) for path in paths])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),ranks=[dict(source=r['source'],
        mean_step_seconds=r['mean_step_seconds'],top_operators=r['operators'][:10]) for r in result['ranks']]),indent=2))


if __name__=='__main__':main()
