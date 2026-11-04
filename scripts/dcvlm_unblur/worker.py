"""Recover DataComp metadata and hash-verified original images; never edits TARs."""
import argparse
import fcntl
import hashlib
import io
import itertools
import json
import os
import socket
import shutil
import tarfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit

import requests
from PIL import Image

from prepare import RUN, save

LOCAL = threading.local()
HOSTS = defaultdict(lambda: threading.BoundedSemaphore(4))


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
        LOCAL.session.headers['User-Agent'] = 'Mozilla/5.0 (compatible; dataset-image-recovery/1.0)'
        LOCAL.session.headers['Accept-Encoding'] = 'identity'
    return LOCAL.session


def download_tar(job, directory, ranges):
    url = f"https://huggingface.co/datasets/mlfoundations/{job['repo']}/resolve/{job['revision']}/{job['path']}"
    parts = directory / 'parts'
    parts.mkdir(exist_ok=True)
    size = job['size']
    step = 4 * 1024 * 1024
    def part(index):
        start, end = index * step, min(size, (index + 1) * step) - 1
        path = parts / f'{index:05d}'
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        temp = path.with_suffix('.tmp')
        for attempt in range(8):
            try:
                position = temp.stat().st_size if temp.exists() else 0
                assert position <= end - start + 1
                if position == end - start + 1:
                    os.replace(temp, path)
                    return
                request_url = url + f'?download=true&dcvlm_request={time.time_ns()}'
                with session().get(request_url, headers={'Range': f'bytes={start + position}-{end}'}, stream=True, timeout=(8, 30)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers.get('Content-Range') == f'bytes {start + position}-{end}/{size}'
                    with temp.open('ab') as output:
                        for block in response.iter_content(65536):
                            output.write(block)
                assert temp.stat().st_size == end - start + 1
                os.replace(temp, path)
                return
            except (requests.RequestException, AssertionError):
                if attempt == 7:
                    raise
                time.sleep(2 + attempt * 2)
    futures = [ranges.submit(part, i) for i in range((size + step - 1) // step)]
    errors = []
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as error:
            errors.append(error)
    if errors:
        raise errors[0]
    assembled = directory / 'source.tar'
    check = hashlib.sha256()
    with assembled.open('wb') as target:
        for index in range((size + step - 1) // step):
            data = (parts / f'{index:05d}').read_bytes()
            check.update(data)
            target.write(data)
    if check.hexdigest() != job['sha256']:
        shutil.rmtree(parts)
        assembled.unlink()
        raise ValueError('Upstream archive SHA256 differs; discarded corrupt cache')
    return assembled


def collect(job, directory, ranges):
    cached = directory / 'metadata.json'
    if cached.exists():
        return json.loads(cached.read_text())
    wanted = set(job['wanted'])
    # Keep only matching image hashes; the metadata can precede or follow images.
    metadata, matched = {}, defaultdict(list)
    source = download_tar(job, directory, ranges)
    with tarfile.open(source, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key, suffix = member.name.rsplit('.', 1)
                if suffix == 'json':
                    metadata[key] = json.load(archive.extractfile(member))
                elif suffix.lower() in ('jpg', 'jpeg', 'png', 'webp'):
                    digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                    if digest in wanted:
                        matched[digest].append(key)
    result = []
    for digest in sorted(wanted):
        keys = matched.get(digest, [])
        key = keys[0] if keys else None
        if not keys:
            result.append(dict(baseline_sha256=digest, status='upstream_image_not_found'))
        elif len({metadata.get(k, {}).get('sha256') for k in keys}) > 1:
            result.append(dict(baseline_sha256=digest, status='ambiguous_original_image'))
        elif key not in metadata or 'face_bboxes' not in metadata[key]:
            result.append(dict(baseline_sha256=digest, status='missing_face_metadata', original_key=key,
                               metadata=metadata.get(key), metadata_member_present=key in metadata))
        else:
            result.append(dict(baseline_sha256=digest, original_key=key,
                               status='identified', metadata=metadata[key]))
    save(cached, result)
    shutil.rmtree(directory / 'parts')
    source.unlink()
    return result


def recover(row, directory):
    digest = row['baseline_sha256']
    output = dict(baseline_sha256=digest, original_key=row.get('original_key'), status=row['status'])
    if row['status'] != 'identified':
        return output
    meta = row['metadata']
    boxes = meta['face_bboxes']
    if boxes is None or not isinstance(boxes, list):
        output['status'] = 'invalid_face_metadata'
        return output
    if not boxes:
        output['status'] = 'no_blur_annotation'
        return output
    output.update(url=meta.get('url'), original_sha256=meta.get('sha256'), face_bboxes=boxes)
    path = directory / 'images' / (digest + '.bin')
    if path.exists():
        data = path.read_bytes()
    else:
        url = meta.get('url', '')
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            output['status'] = 'invalid_url'
            return output
        try:
            with HOSTS[parsed.hostname]:
                with session().get(url, stream=True, timeout=(10, 25)) as response:
                    if response.status_code != 200:
                        output.update(status='http_error', http_status=response.status_code)
                        return output
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 32 * 1024 * 1024:
                            output['status'] = 'image_exceeds_32mib'
                            return output
                        chunks.append(chunk)
                    data = b''.join(chunks)
        except requests.RequestException as error:
            output.update(status='network_error', error_type=type(error).__name__)
            return output
    actual = hashlib.sha256(data).hexdigest()
    if actual != meta.get('sha256'):
        output.update(status='original_sha256_mismatch', downloaded_sha256=actual)
        return output
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            output.update(width=im.width, height=im.height, format=im.format)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        output.update(status='decode_error', error_type=type(error).__name__)
        return output
    if not path.exists():
        temp = path.with_suffix('.tmp')
        temp.write_bytes(data)
        os.replace(temp, path)
    output.update(status='restored', path=str(path), size=len(data), sha256=actual)
    return output


def process(job, downloads, ranges):
    name = job['path'].rsplit('/', 1)[-1]
    directory = RUN / 'upstream' / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('w') as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (directory / 'DONE.json').exists():
            return
        save(directory / 'ACTIVE.json', dict(host=socket.gethostname(), pid=os.getpid(), started=time.time()))
        (directory / 'images').mkdir(exist_ok=True)
        try:
            rows = collect(job, directory, ranges)
            previous = {}
            log = directory / 'results.jsonl'
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        item = json.loads(line)
                        previous[item['baseline_sha256']] = item
                    except json.JSONDecodeError:
                        pass  # A killed process can leave only the last line incomplete.
            retryable = {'network_error', 'http_error'}
            if log.exists() and log.stat().st_size:
                with log.open('rb+') as stream:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b'\n':
                        stream.write(b'\n')
            with log.open('a') as stream:
                for attempt in range(3):
                    pending = [row for row in rows if row['baseline_sha256'] not in previous or
                               previous[row['baseline_sha256']]['status'] in retryable]
                    futures = [downloads.submit(recover, row, directory) for row in pending]
                    for future in as_completed(futures):
                        result = future.result()
                        result['attempt'] = attempt + 1
                        previous[result['baseline_sha256']] = result
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                        stream.flush()
                    if not pending:
                        break
                    if attempt < 2:
                        time.sleep(5)
            assert len(previous) == len(rows)
            counts = dict(Counter(x['status'] for x in previous.values()))
            save(directory / 'results.json', list(previous.values()))
            save(directory / 'DONE.json', dict(counts=counts, expected=len(rows), ended=time.time(),
                                               upstream={k: v for k, v in job.items() if k != 'wanted'}))
            print(name, counts, flush=True)
        except Exception as error:
            save(directory / 'ERROR.json', dict(error_type=type(error).__name__, ended=time.time()))
            print(name, 'ERROR', type(error).__name__, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--scan-threads', type=int, default=2)
    parser.add_argument('--download-threads', type=int, default=128)
    args = parser.parse_args()
    worker_lock = RUN / f'worker_{args.worker:02d}.lock'
    with worker_lock.open('w') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code_sha = hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
        state = dict(worker=args.worker, pid=os.getpid(), host=socket.gethostname(), code_sha256=code_sha, phase='download')
        def heartbeat():
            while True:
                save(RUN / f'worker_{args.worker:02d}.heartbeat.json', dict(state, updated=time.time()))
                time.sleep(20)
        threading.Thread(target=heartbeat, daemon=True).start()
        while not (RUN / 'jobs.json').exists():
            time.sleep(10)
        jobs = json.loads((RUN / 'jobs.json').read_text())
        # Rotate the queue so each instance immediately begins at a different shard.
        offset = len(jobs) * args.worker // 32
        jobs = jobs[offset:] + jobs[:offset]
        with ThreadPoolExecutor(max_workers=args.download_threads) as downloads:
          with ThreadPoolExecutor(max_workers=16) as ranges:
            with ThreadPoolExecutor(max_workers=args.scan_threads) as scans:
                for sweep in itertools.count():
                    list(scans.map(lambda job: process(job, downloads, ranges), jobs))
                    remaining = sum(not (RUN / 'upstream' / j['path'].rsplit('/', 1)[-1] / 'DONE.json').exists() for j in jobs)
                    save(RUN / f'worker_{args.worker:02d}.json', dict(remaining=remaining, sweep=sweep, updated=time.time()))
                    if not remaining:
                        break
                    time.sleep(30)
        state['phase'] = 'waiting_for_apply'
        while not (RUN / 'APPLY_READY.json').exists():
            time.sleep(10)
        state['phase'] = 'apply'
        from pipeline import apply_worker
        apply_worker(args.worker)
        state['phase'] = 'finished'
        save(RUN / f'worker_{args.worker:02d}.heartbeat.json', dict(state, updated=time.time()))


if __name__ == '__main__':
    main()
