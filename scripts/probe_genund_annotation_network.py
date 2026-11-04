"""Bounded public metadata transfer probes; no auth and no image downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlparse
import requests

OUT = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1/annotations')
SCALE = 'datasets/long-xing1/ScaleCap-450k/resolve/c1de9c7fffa494a34c71adb7eced9bcc0bb0d957/ScaleCap450k.json'
FINE = 'datasets/HuggingFaceM4/FineVision/resolve/3c380a731a3429c1d04693d6ec16d7e683def84c/CoSyn_400k_circuit/train-00000-of-00001.parquet'
URLS = [('hf_scale_download', 'https://huggingface.co/'+SCALE+'?download=true', 'bytes=0-16383'),
        ('mirror_scale', 'https://hf-mirror.com/'+SCALE, 'bytes=0-16383'),
        ('hf_fine_footer', 'https://huggingface.co/'+FINE, 'bytes=-65536'),
        ('mirror_fine_footer', 'https://hf-mirror.com/'+FINE, 'bytes=-65536'),
        ('viewer_scale', 'https://datasets-server.huggingface.co/rows?dataset=long-xing1%2FScaleCap-450k&config=default&split=train&offset=0&length=2', None),
        ('viewer_fine', 'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4%2FFineVision&config=CoSyn_400k_circuit&split=train&offset=0&length=2', None)]


def probe(item):
    label,url,byte_range = item; result = {'label':label}
    try:
        with requests.get(url,headers={'Range':byte_range} if byte_range else {},timeout=(10,25),stream=True) as response:
            result.update(status=response.status_code,host=urlparse(response.url).netloc,
                          content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'))
            data = next(response.iter_content(65536), b'')
            (OUT/(label+'.sample')).write_bytes(data)
            result['bytes_read'] = len(data)
            result['prefix'] = data[:100].decode('utf-8',errors='replace') if label.startswith('viewer') else None
    except Exception as exc: result.update(error_type=type(exc).__name__,error=str(exc)[:300])
    return result


with ThreadPoolExecutor(max_workers=6) as pool: result = list(pool.map(probe,URLS))
(OUT/'network_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
