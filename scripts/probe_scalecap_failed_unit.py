import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from corpus_pipeline.production.common import atomic_json
from corpus_pipeline.production.genund_collections import iter_json_records,MAX_IMAGE
RUN=Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1')
plan=json.loads((RUN/'ingest_units/scalecap/PLAN.json').read_text()); failures={p.stem.split('.')[0] for p in (RUN/'normalized_sources/ScaleCap450k_image/claims').glob('*.error.json')}; reports=[]
for unit in plan['units']:
    if unit['unit_id'] not in failures: continue
    for _,wrapped in iter_json_records(unit['path']):
        for image in wrapped['record']['images']:
            name=image if image.startswith('images/') else 'images/'+image
            path=Path(unit['root'])/'ScaleCap450k_image'/name
            if path.exists():
                size=path.stat().st_size
                if size==0 or size>MAX_IMAGE: reports.append({'unit':unit['unit_id'],'row_index':wrapped['row_index'],'path':str(path),'size':size})
atomic_json(RUN/'scalecap_problem_image_sizes.json',reports); print(json.dumps(reports))
