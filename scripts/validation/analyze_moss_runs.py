"""Collect completed real-data runs; never treat capacity slots as corpus tokens."""
import argparse
import json
from pathlib import Path
import statistics

DOCUMENTS=119_010_208
IMAGE_OCCURRENCES=108_026_384


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'.runs/qwen35-moss')
    args=parser.parse_args();runs=[];excluded=[];failed=[];excluded_details=[]
    for path in sorted(args.root.glob('r*/*/summary.json')):
        row=json.loads(path.read_text())
        if row.get('parameters')!=14_492_122_352:continue
        name=str(path.parent.relative_to(args.root))
        if row.get('profiling') or row.get('validation_probes') or row.get('fake_latents'):
            excluded.append(name);continue
        nodes={p.stem.removeprefix('node') for p in (path.parent/'rendezvous').glob('node*.json')}
        exits={p.stem.removeprefix('exit'):json.loads(p.read_text())['returncode']
               for p in path.parent.glob('exit*.json')}
        if not nodes or set(exits)!=nodes or any(exits.values()):
            excluded.append(name)
            excluded_details.append(dict(run=name,reason='Missing or unsuccessful node completion',
                                         expected_nodes=sorted(nodes),returncodes=exits))
            continue
        steps=[json.loads(line) for line in (path.parent/'steps.jsonl').read_text().splitlines()]
        measured={r['step'] for r in steps if not r['warmup']}
        documents=sum(r.get('documents_completed',0) for r in steps if r['step'] in measured)
        if not documents:
            for trace in path.parent.glob('consumed_rank_*.jsonl'):
                if int(trace.stem.split('_')[-1])%row['tp']:continue
                for entry in map(json.loads,trace.read_text().splitlines()):
                    if entry['step'] in measured:
                        documents+=sum(d['completed'] for receipt in entry['receipts'] for d in receipt['documents'])
        docs_per_second=documents/row['wall_seconds']
        value=dict(run=name,measured_updates=len(measured),wall_seconds=row['wall_seconds'],
            tp=row['tp'],dp=row['dp'],optimizer_instances=row['optimizer_instances'],
            images_per_second=row['images_per_second'],documents_per_second=docs_per_second,
            peak_memory_gib=row['peak_memory_gib'],work_budget=row.get('work_budget',0),
            image_epoch_days=IMAGE_OCCURRENCES/row['images_per_second']/86400,
            document_epoch_days=DOCUMENTS/docs_per_second/86400,
            host_wait_seconds=row['data_wait_seconds'],device_wait_seconds=row['data_device_wait_seconds'])
        value['measured_images']=sum(r['images_consumed'] for r in steps if r['step'] in measured)
        value['measured_documents']=documents
        value['successful_nodes']=len(exits)
        if all('active_microbatches' in r for r in steps if r['step'] in measured):
            value['mean_actual_global_microbatches']=statistics.mean(
                r['active_microbatches'] for r in steps if r['step'] in measured)
        metadata=json.loads((path.parent/'run.json').read_text())
        arguments=metadata['arguments']
        model=metadata['model']
        value.update(seed=arguments['seed'],ce_chunk=arguments['ce_chunk'],grad_accum=row['grad_accum'],
            generation_max_pixels=metadata['data'].get('generation_max_pixels',model['generation_max_pixels']),
            vision_max_pixels=model['vision_max_pixels'],
            freeze_vision=model.get('freeze_vision',False),
            freeze_vision_merger=model.get('freeze_vision_merger',False),
            trainable_parameters=row.get('trainable_parameters',row['parameters']),
            frozen_parameters=row.get('frozen_parameters',0),
            moss_fused_pointwise=metadata['kernel_options'].get('moss_fused_pointwise','0'),
            mc2_row=metadata['kernel_options'].get('mc2_row','0'))
        work={step:[] for step in measured}
        for trace in path.parent.glob('timing_rank_*.jsonl'):
            if int(trace.stem.split('_')[-1])%row['tp']:continue
            for entry in map(json.loads,trace.read_text().splitlines()):
                if entry['step'] in measured and 'work' in entry:work[entry['step']].append(entry['work'])
        full=[values for values in work.values() if len(values)==row['dp'] and sum(values)>0]
        if full:
            value['scheduling_work_max_over_mean']=statistics.mean(max(v)/statistics.mean(v) for v in full)
            value['scheduling_work_min_over_mean']=statistics.mean(min(v)/statistics.mean(v) for v in full)
            value['scheduling_work_note']='Heuristic pack-cost balance, not measured FLOPs or device utilization.'
        runs.append(value)
    for path in sorted(args.root.glob('r*/*/exit0.json')):
        exits=[json.loads(p.read_text())['returncode'] for p in path.parent.glob('exit*.json')]
        if any(exits):failed.append(dict(run=str(path.parent.relative_to(args.root)),returncodes=exits))
    result=dict(corpus_documents=DOCUMENTS,corpus_image_occurrences=IMAGE_OCCURRENCES,
        required_images_per_second_for_five_days=IMAGE_OCCURRENCES/(5*86400),
        required_documents_per_second_for_five_days=DOCUMENTS/(5*86400),
        caveat='Epoch figures extrapolate the measured stream and exclude downtime; short runs do not certify full-corpus runtime. Compare resolution and trainable scope before attributing speed differences to infrastructure.',
        completed_runs=runs,excluded_from_throughput=excluded,excluded_details=excluded_details,failed_stages=failed)
    out=args.root/'analysis';out.mkdir(exist_ok=True)
    (out/'training_comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
