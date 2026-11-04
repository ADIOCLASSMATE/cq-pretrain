"""One real-image transport/function smoke; not an AI-detection quality test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import httpx
from corpus_pipeline.production.client import image_payload


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.chmod(0o600); temporary.replace(path)


def run(input_path):
    request = json.loads(input_path.read_text())
    output = input_path.parent / 'RESULT.json'
    result = {'schema': 'corpus.native_serving.basic_smoke.v1', 'passed': False,
        'qualification_kind': 'basic_smoke', 'quality_canary_performed': False,
        'model': request['model'], 'revision': request['revision'], 'precision': request['precision'],
        'processor_revision': request['processor_revision'], 'runtime_revision': request['runtime_revision'],
        'incarnation': request['incarnation'], 'expected_boots': request['expected_boots'],
        'request_count': 1, 'started_at': time.time(),
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}
    try:
        asset = request['asset']
        image, metadata = image_payload(asset)
        payload = {'model': request['model'], 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe the main visible objects in this image in one short sentence.'}, image]}],
            'max_tokens': 192, 'temperature': 0.1, 'stream': False, **request['request_overrides']}
        key = Path(request['api_key_file']).read_text().strip()
        started = time.monotonic()
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(180, connect=10)) as client:
            response = client.post(request['base_url'].rstrip('/') + '/chat/completions', json=payload,
                                   headers={'Authorization': 'Bearer ' + key})
        result.update(http_status=response.status_code, latency_seconds=time.monotonic()-started,
                      boot_id=response.headers.get('x-umm-boot'), image_input=metadata)
        response.raise_for_status()
        body = response.json(); choice = body['choices'][0]
        content = choice['message'].get('content')
        assert body['model'] == request['model'], 'Response model changed'
        assert result['boot_id'] in request['expected_boots'], 'Response boot changed'
        assert choice['finish_reason'] == 'stop' and isinstance(content, str) and content.strip(), 'Response is incomplete'
        assert body.get('usage', {}).get('prompt_tokens', 0) > 0, 'Missing input usage'
        result.update(passed=True, response_model=body['model'], finish_reason=choice['finish_reason'],
                      response_text=content, usage=body.get('usage', {}))
    except Exception as error:
        result['error_type'] = type(error).__name__
    result['finished_at'] = time.time()
    save(output, result)
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.input) else 1)
