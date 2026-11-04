"""One standard OpenAI SSE real-image functional smoke, entirely outside Serving."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import httpx
from corpus_pipeline.production.client import image_payload


def run(path):
    request=json.loads(path.read_text());result={k:request[k] for k in ('model','revision','precision','processor_revision',
        'runtime_revision','incarnation','platform_instance_id','launch_contract_sha256')}
    result.update(schema='corpus.official_native.basic_smoke.v1',passed=False,qualification_kind='basic_smoke',
        response_boot_attested=False,attestation_scope='platform_instance_and_registry',request_count=1,started_at=time.time(),
        quality_canary_performed=False,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        image,metadata=image_payload(request['asset'])
        payload={'model':request['model'],'messages':[{'role':'user','content':[
            {'type':'text','text':'Describe the main visible objects in this image in one short sentence.'},image]}],
            'max_tokens':192,'temperature':0.1,'stream':True,'stream_options':{'include_usage':True},**request['request_overrides']}
        headers={'Authorization':'Bearer '+Path(request['api_key_file']).read_text().strip()}
        text=[];finish=None;done=False;usage={};start=time.monotonic()
        with httpx.Client(trust_env=False,timeout=httpx.Timeout(180,connect=10)) as client:
            with client.stream('POST',request['base_url']+'/chat/completions',json=payload,headers=headers) as response:
                result['http_status']=response.status_code;response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data:'):continue
                    data=line[5:].strip()
                    if data=='[DONE]':done=True;break
                    if not data:continue
                    item=json.loads(data)
                    assert item.get('model')==request['model'],'Response model changed'
                    if item.get('usage'):usage=item['usage']
                    for choice in item.get('choices',[]):
                        if choice.get('finish_reason'):finish=choice['finish_reason']
                        part=choice.get('delta',{}).get('content')
                        if part:text.append(part)
        content=''.join(text)
        assert done and finish=='stop' and content.strip(),'Incomplete standard stream'
        assert usage.get('prompt_tokens',0)>0,'Missing usage'
        result.update(passed=True,finish_reason=finish,response_text=content,usage=usage,image_input=metadata,
            latency_seconds=time.monotonic()-start,standard_sse_done=done)
    except Exception as error:result['error_type']=type(error).__name__
    result['finished_at']=time.time();target=path.parent/'RESULT.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.chmod(0o600);tmp.replace(target)
    return result['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True,type=Path)
    raise SystemExit(0 if run(parser.parse_args().input) else 1)
