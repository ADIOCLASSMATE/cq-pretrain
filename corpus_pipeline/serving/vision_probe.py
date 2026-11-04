"""Authenticated two-image canary and bounded concurrency probe, with full receipts."""
from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import secrets
import time

import httpx
from PIL import Image, ImageDraw, ImageFont


def card():
    digits = ''.join(str(secrets.randbelow(10)) for _ in range(6))
    color = secrets.choice(['red', 'blue', 'green', 'orange'])
    image = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 82)
    draw.text((256, 135), digits, font=font, fill='black', anchor='mm')
    draw.rectangle((65, 270, 447, 450), fill=color)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue(), {'digits': digits, 'color': color}


def parse_json(text):
    value = text.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(value)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    token = os.environ[args.api_key_env]
    report = {'model': args.model, 'base_url': args.base_url, 'revision': args.revision,
              'expected_boots': args.expected_boot,
              'request_overrides': json.loads(args.request_overrides_json),
              'started_at': time.time(), 'proxy_used': False, 'passed': False, 'requests': [], 'tiers': []}
    report_path = args.output / 'result.json'
    def save():
        temp = report_path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(report_path)
    save()
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(900, connect=10),
                headers={'Authorization': 'Bearer ' + token},
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)) as client:
            models = await client.get(args.base_url + '/models')
            models.raise_for_status()
            assert any(x['id'] == args.model for x in models.json()['data']), 'Wrong served model'
            invalid = await client.get(args.base_url + '/models', headers={'Authorization': 'Bearer invalid-canary'})
            assert invalid.status_code in (401, 403), 'Authentication not enforced'
            report['authentication_passed'] = True

            async def request_one(index):
                raw1, expected1 = card()
                raw2, expected2 = card()
                (args.output / f'{index:03}-first.png').write_bytes(raw1)
                (args.output / f'{index:03}-second.png').write_bytes(raw2)
                content = [{'type': 'text', 'text': 'Inspect both images. Read the six digits exactly, including leading zeros, and identify the rectangle color. Return only JSON with keys first and second in image order; each value has string keys digits and color.'}]
                content += [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}} for raw in (raw1, raw2)]
                body = {'model': args.model, 'temperature': 0, 'max_tokens': 512,
                        'chat_template_kwargs': {'enable_thinking': False},
                        'messages': [{'role': 'user', 'content': content}]}
                overrides=json.loads(args.request_overrides_json)
                assert set(overrides)<= {'chat_template_kwargs','mm_processor_kwargs'}, 'Unsupported probe request override'
                body.update(overrides)
                (args.output / f'{index:03}-request.json').write_text(json.dumps(body, ensure_ascii=False) + '\n')
                start = time.monotonic()
                response = await client.post(args.base_url + '/chat/completions', json=body)
                (args.output / f'{index:03}-response.json').write_text(response.text + '\n')
                response.raise_for_status()
                payload = response.json()
                assert payload.get('model') == args.model, 'Response model differs from requested model'
                actual = parse_json(payload['choices'][0]['message']['content'])
                expected = {'first': expected1, 'second': expected2}
                item = {'index': index, 'seconds': time.monotonic()-start, 'expected': expected,
                        'actual': actual, 'passed': actual == expected, 'usage': payload.get('usage'),
                        'replica': response.headers.get('x-umm-replica')}
                item['boot_id'] = response.headers.get('x-umm-boot') or response.headers.get('x-umm-boot-id')
                if args.expected_boot:
                    assert item['boot_id'] in args.expected_boot, 'Response belongs to an unexpected process incarnation'
                report['requests'].append(item)
                save()
                assert item['passed'], 'Image decoding/order/content canary failed'
                return item

            index = 0
            for concurrency in args.concurrency:
                start = time.monotonic()
                items = await asyncio.gather(*(request_one(index+i) for i in range(concurrency)))
                seconds = time.monotonic()-start
                report['tiers'].append({'concurrency': concurrency, 'requests': len(items), 'seconds': seconds,
                     'requests_per_second': len(items)/seconds,
                     'completion_tokens_per_second': sum((x.get('usage') or {}).get('completion_tokens', 0) for x in items)/seconds})
                index += concurrency
                save()
            replicas = {item['replica'] for item in report['requests'] if item.get('replica')}
            assert all(item.get('replica') for item in report['requests']), 'Missing backend replica provenance'
            assert len(replicas) == args.expected_replicas, 'Not every expected local replica was verified'
            report['replicas_verified'] = sorted(replicas)
            report['passed'] = True
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)[:2000]
        if isinstance(exc, httpx.HTTPStatusError):
            report['error_status'] = exc.response.status_code
    finally:
        report['finished_at'] = time.time()
        save()
    print(json.dumps({'passed': report['passed'], 'requests': len(report['requests']), 'result': str(report_path)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--expected-replicas', type=int, default=1)
    parser.add_argument('--api-key-env', default='CORPUS_INFERENCE_API_KEY')
    parser.add_argument('--expected-boot', action='append', default=[])
    parser.add_argument('--request-overrides-json', default='{}')
    args = parser.parse_args()
    if any(n < 1 or n > 32 for n in args.concurrency):
        parser.error('concurrency must be 1..32')
    raise SystemExit(asyncio.run(run(args)))
