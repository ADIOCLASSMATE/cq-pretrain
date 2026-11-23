"""Real pinned source-file recovery for single/multi-image binding qualification."""
import json
from pathlib import Path
import sys
import traceback

from finevision_recovery_queue import inventory,recover

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from corpus_pipeline.production.common import atomic_json
from corpus_pipeline.production.genund_collections import Storage,finevision_record,iter_json_records

RUN=Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1')
plan=json.loads((RUN/'finevision_recovery/PLAN.json').read_text()); binding=json.loads((RUN/'DATASET_BINDING.json').read_text())
area=RUN/'finevision_recovery_canary'; reports=[]
for subset in ['nlvr2','CoSyn_400k_circuit','spot_the_diff']:
    task=next(task for task in plan['tasks'] if task['subset']==subset)
    try:
        metadata=inventory(task,area)
        task={**task,'row_start':0,'image_start':0,'rows':metadata['rows'],'images':metadata['images']}
        result=recover(task,plan,area)
        examples=[]
        with Storage(plan['root'],binding['dataset_id']) as store:
            for index,row in iter_json_records(result['path']):
                record=next(finevision_record({'source_shard':task['path'],'subset':subset},row,index,store))
                examples.append({'images':len(record['source_image_refs']),'segments':len(record['raw']['segments']),
                                 'issues':record['metadata']['issues'],'route':record['raw']['task_route']})
                if len(examples)>=3: break
        reports.append({'subset':subset,'complete':True,'recovery':result,'examples':examples})
    except Exception as exc:
        reports.append({'subset':subset,'complete':False,'error_type':type(exc).__name__,'error':str(exc)[:1000],
                        'traceback':traceback.format_exc()})
    atomic_json(area/'REPORT.json',reports)
print(json.dumps(reports))
