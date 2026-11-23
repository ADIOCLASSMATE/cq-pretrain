"""Full official annotation path coverage, including explicit raw filename bytes."""
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from corpus_pipeline.production.common import atomic_json
from corpus_pipeline.production.genund_collections import iter_json_records,relative

RUN=Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1')
ROOT=Path('/inspire/dataset/gen-und-mm-pretrain/01'); references=Counter(); rows=0
for index,row in iter_json_records(RUN/'annotations/ScaleCap-450k/ScaleCap450k.json'):
    rows+=1
    for image in row['images']:
        name=relative(image)
        if not name.startswith('images/'): name='images/'+name
        references['ScaleCap450k_image/'+name]+=1
def check(name): return name,(ROOT/name).is_file()
with ThreadPoolExecutor(max_workers=32) as pool: missing=[name for name,exists in pool.map(check,references) if not exists]
odd=[]
for leaf in ['share4v_image','laion_filter_image']:
    directory=os.fsencode(ROOT/'ScaleCap450k_image/images'/leaf)
    for entry in os.scandir(directory):
        try: entry.name.decode('utf8'); continue
        except UnicodeDecodeError: pass
        matches=[]
        for codec in ['cp437','cp936','gb18030','latin1']:
            try: decoded=entry.name.decode(codec)
            except UnicodeDecodeError: continue
            name='ScaleCap450k_image/images/'+leaf+'/'+decoded
            if name in references: matches.append({'codec':codec,'annotation_path':name,'references':references[name]})
        odd.append({'leaf':leaf,'name_bytes_b64':base64.b64encode(entry.name).decode(),'annotation_encoding_candidates':matches})
result={'complete':True,'annotation_records':rows,'image_references':sum(references.values()),
        'distinct_image_paths':len(references),'existing_distinct_paths':len(references)-len(missing),
        'missing_distinct_paths':missing,'missing_references':sum(references[name] for name in missing),
        'non_utf8_source_files':odd,'non_utf8_file_count':len(odd),'source_modified':False,
        'encoding_candidates_are_not_binding_authority':True}
atomic_json(RUN/'scalecap_full_path_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in {'missing_distinct_paths','non_utf8_source_files'}}))
