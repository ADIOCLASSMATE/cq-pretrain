"""Report durable bytes and interval throughput, never count old rates twice."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus_pipeline.production.common import atomic_json

parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args()
area=Path(args.run)/'finevision_recovery'; plan=json.loads((area/'PLAN.json').read_text()); now=time.time()
complete={p.parent.name:json.loads(p.read_text()) for p in (area/'recovered').glob('*/COMPLETE.json')}
progress={p.parent.name:json.loads(p.read_text()) for p in (area/'recovered').glob('*/DOWNLOAD_PROGRESS.json')}
available={key:value.get('available_bytes',0) for key,value in progress.items()}
for key,value in complete.items(): available[key]=value['verified_upstream_bytes']
heartbeat=[json.loads(path.read_text()) for path in (area/'workers').glob('*.heartbeat.json')]
result={'time':now,'inventory_complete':len(list((area/'inventory').glob('*/COMPLETE.json'))),
        'local_sha_indices_complete':len(list((area/'local_assets').glob('*/COMPLETE.json'))),
        'source_files':plan['task_count'],'source_bytes':plan['source_bytes'],
        'recovered_files_complete':len(complete),'durable_download_bytes':sum(available.values()),
        'verified_upstream_bytes':sum(value['verified_upstream_bytes'] for value in complete.values()),
        'records_verified':sum(value['records'] for value in complete.values()),
        'image_slots_verified':sum(value['image_slots'] for value in complete.values()),
        'recovered_missing_image_slots':sum(value['recovered_missing_image_slots'] for value in complete.values()),
        'active_instances':sum(now-value['updated_at']<90 for value in heartbeat),
        'errors':len(list(area.glob('*/*/ERROR.json')))}
path=area/'THROUGHPUT_STATUS.json'
if path.exists():
    prior=json.loads(path.read_text()); elapsed=now-prior['time']; delta=result['durable_download_bytes']-prior['durable_download_bytes']
    if elapsed>0 and delta>=0:
        result['interval_seconds']=elapsed; result['interval_MB_per_second']=delta/elapsed/1e6
        if delta: result['estimated_remaining_network_hours']=(plan['source_bytes']-result['durable_download_bytes'])/(delta/elapsed)/3600
atomic_json(path,result); print(json.dumps(result))
