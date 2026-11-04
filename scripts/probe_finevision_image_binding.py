"""Compare original first-row-group image bytes with explicit local row candidates."""
import hashlib
import io
import json
from pathlib import Path
import sys
from PIL import Image
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent))
from recover_genund_annotation_ranges import RangeFile,ROOT

info=json.loads((ROOT/'FineVision/hub_info.json').read_text()); results=[]
for subset in ['CoSyn_400k_circuit','nlvr2','spot_the_diff']:
    result={'subset':subset,'comparisons':[]}
    try:
        item=next(x for x in info['siblings'] if x['rfilename'].startswith(subset+'/') and x['rfilename'].endswith('.parquet'))
        url=f"https://huggingface.co/datasets/HuggingFaceM4/FineVision/resolve/{info['sha']}/{item['rfilename']}?download=true"
        with RangeFile(url,item['size']) as stream:
            reader=pq.ParquetFile(stream); table=reader.read_row_group(0,columns=['images'])
            image_offsets=[]; image_cursor=0
            for item_images in table['images'].to_pylist():
                image_offsets.append(image_cursor); image_cursor+=len(item_images)
            for row_index in [0,1,2,17,table.num_rows-1]:
                images=table['images'][row_index].as_py(); local=Path('/inspire/dataset/gen-und-mm-pretrain/01/FineVision')/subset/f'{row_index}.jpg'
                record={'row_index':row_index,'original_image_count':len(images),'local_path':str(local),'local_exists':local.is_file()}
                record['all_image_binding_candidates']=[]
                for image_index, source_image in enumerate(images):
                    source_sha=hashlib.sha256(source_image['bytes']).hexdigest()
                    candidates=[f'{row_index}.jpg',f'{row_index}_{image_index}.jpg',f'{image_offsets[row_index]+image_index}.jpg']
                    matches=[]
                    for candidate in dict.fromkeys(candidates):
                        candidate_path=local.parent/candidate
                        if candidate_path.is_file() and hashlib.sha256(candidate_path.read_bytes()).hexdigest()==source_sha: matches.append(candidate)
                    record['all_image_binding_candidates'].append({'image_index':image_index,'source_sha256':source_sha,'matches':matches})
                if local.is_file():
                    local_blob=local.read_bytes(); record['local_sha256']=hashlib.sha256(local_blob).hexdigest()
                    with Image.open(io.BytesIO(local_blob)) as im: record['local_size']=im.size; record['local_format']=im.format
                    if len(images)==1:
                        blob=images[0]['bytes']; record['source_sha256']=hashlib.sha256(blob).hexdigest(); record['byte_equal']=blob==local_blob
                        with Image.open(io.BytesIO(blob)) as im:
                            record['source_size']=im.size; record['source_format']=im.format
                            output=io.BytesIO(); im.convert('RGB').save(output,format='JPEG')
                            record['pil_rgb_jpeg_default_exact']=output.getvalue()==local_blob
                result['comparisons'].append(record)
            result.update(network_bytes=stream.bytes_read,range_requests=stream.requests,source=item)
    except Exception as exc: result.update(error_type=type(exc).__name__,error=str(exc)[:500])
    results.append(result); (ROOT/'FineVision/image_binding_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False))
