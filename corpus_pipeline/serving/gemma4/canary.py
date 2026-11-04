"""Real DCVLM vision + JSON-schema + ordered multi-image endpoint checks.

Example: inspire account api-key run dev -- .venv/bin/python .../canary.py
--base-url https://<service>/v1 --output .../canary/<new_run>
No image authenticity label is inferred from these functional checks.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import threading

import httpx

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
IMAGES = ROOT.parents[1] / 'runs' / 'detector_calibration_20261022' / 'images'
CUPS = '63304a67a6dfe708962ec1da1c7a03b7a3d181651973581ff04e723726c68500.jpg'
CLEVR = 'a2eaef0691206e59202077c6813f302a7fc9dcf90a5f454b6ee558471ee9ffe2.png'


def image_part(name):
    data = (IMAGES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == name.split('.')[0]
    mime = 'image/png' if name.endswith('.png') else 'image/jpeg'
    return {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--expect-replicas', type=int, choices=[1, 4], default=1)
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ['INF_API_KEY']
    args.output.mkdir(parents=True, exist_ok=False)
    client = httpx.Client(trust_env=False, timeout=httpx.Timeout(900, connect=20),
                          headers={'Authorization': 'Bearer ' + key})
    health_url = args.base_url.rstrip('/').removesuffix('/v1') + '/health'
    health_before_response = client.get(health_url)
    health_before_response.raise_for_status()
    health_before = health_before_response.json()
    assert health_before.get('model_ready') and health_before.get('boot_id')
    cups, clevr = image_part(CUPS), image_part(CLEVR)
    cases = [
        ('single_photo', [cups, {'type': 'text', 'text': 'Describe the main visible objects in this image in one factual English sentence. Do not infer provenance or hidden details.'}], None),
        ('structured_clevr', [clevr, {'type': 'text', 'text': 'Inspect the image. Count the visible objects and identify their shapes. Return JSON matching the schema.'}],
         {'type': 'object', 'properties': {'object_count': {'type': 'integer'}, 'shapes': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['object_count', 'shapes'], 'additionalProperties': False}),
        ('ordered_multi_image', [cups, clevr, {'type': 'text', 'text': 'These are two different images in order. Return concise descriptions under first_image and second_image. Keep the image order and do not mix their objects.'}],
         {'type': 'object', 'properties': {'first_image': {'type': 'string'}, 'second_image': {'type': 'string'}}, 'required': ['first_image', 'second_image'], 'additionalProperties': False}),
    ]
    reports = []
    for name, content, schema in cases:
        payload = {'model': CONFIG['model_name'], 'messages': [{'role': 'user', 'content': content}],
                   'max_tokens': 4096 if args.thinking else 1024, 'temperature': 0,
                   'chat_template_kwargs': {'enable_thinking': args.thinking}}
        if schema:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}
        started = time.monotonic()
        response = client.post(args.base_url.rstrip('/') + '/chat/completions', json=payload)
        report = {'case': name, 'status': response.status_code, 'seconds': round(time.monotonic() - started, 3),
                  'model': CONFIG['model_name'], 'revision': CONFIG['revision'], 'enable_thinking': args.thinking,
                  'image_sha256': [CUPS.split('.')[0]] if name == 'single_photo' else
                                  [CLEVR.split('.')[0]] if name == 'structured_clevr' else
                                  [CUPS.split('.')[0], CLEVR.split('.')[0]],
                  'schema_requested': schema, 'replica': response.headers.get('x-umm-replica'),
                  'boot_id': response.headers.get('x-umm-boot-id')}
        try:
            body = response.json()
        except ValueError:
            body = {'error': 'non_json_http_response'}
        if response.status_code == 200 and body.get('choices'):
            choice = body['choices'][0]
            message = choice['message']
            report.update(content=message.get('content'), finish_reason=choice.get('finish_reason'),
                          response_model=body.get('model'), model_matches_request=body.get('model') == CONFIG['model_name'],
                          usage=body.get('usage'), reasoning_present=bool(message.get('reasoning') or message.get('reasoning_content')))
            if schema:
                try:
                    result = json.loads(message['content'])
                    report['parsed_json'] = result
                    report['schema_keys_match'] = set(result) == set(schema['required'])
                    if name == 'structured_clevr':
                        report['expected_count_matches'] = result.get('object_count') == 3
                except (ValueError, TypeError):
                    report['parsed_json'] = None
                    report['schema_keys_match'] = False
        else:
            report['error'] = body.get('error', 'unexpected_response')
        (args.output / (name + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.expect_replicas == 4:
        barrier = threading.Barrier(4)
        def replica_probe(index):
            barrier.wait(timeout=30)
            response = client.post(args.base_url.rstrip('/') + '/chat/completions', json={
                'model': CONFIG['model_name'], 'messages': [{'role': 'user', 'content': [clevr, {
                    'type': 'text', 'text': 'How many objects are visible? Answer with one integer only.'}]}],
                'max_tokens': 64, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
            body = response.json()
            message = body.get('choices', [{}])[0].get('message', {})
            return {'case': 'parallel_replica_' + str(index), 'status': response.status_code,
                    'replica': response.headers.get('x-umm-replica'), 'content': message.get('content'),
                    'boot_id': response.headers.get('x-umm-boot-id'),
                    'response_model': body.get('model'), 'model_matches_request': body.get('model') == CONFIG['model_name'],
                    'usage': body.get('usage'), 'reasoning_present': bool(message.get('reasoning') or message.get('reasoning_content'))}
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(replica_probe, range(4)))
        (args.output / 'parallel_replicas.json').write_text(json.dumps(parallel, indent=2) + '\n')
        reports.extend(parallel)
        print(json.dumps({'parallel_replica_results': parallel}), flush=True)
    health_after_response = client.get(health_url)
    health_after_response.raise_for_status()
    health_after = health_after_response.json()
    receipt = {'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'base_url': args.base_url,
               'cases': reports, 'all_http_success': all(r['status'] == 200 for r in reports),
               'health_before': health_before, 'health_after': health_after,
               'all_boots_match': all(r.get('boot_id') == health_before['boot_id'] == health_after.get('boot_id') for r in reports),
               'all_response_models_match': all(r.get('model_matches_request') is True for r in reports),
               'expected_replicas': args.expect_replicas,
               'parallel_distinct_replicas': len({r.get('replica') for r in reports if r['case'].startswith('parallel_') and r.get('replica')}),
               'requires_visual_review': True, 'published_training': False}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    client.close()


if __name__ == '__main__':
    main()
