"""Look for byte-identical original images already present in the baseline."""
import hashlib
import fcntl
import io
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from prepare import NATIVE, ROOT, RUN, save


def main():
    started = time.time()
    wanted = set()
    for path in (RUN / 'upstream').glob('*/results.json'):
        for row in json.loads(path.read_text()):
            if row.get('face_bboxes') and row['status'] != 'restored' and row.get('original_sha256'):
                wanted.add(row['original_sha256'])
    hashes = pa.array(sorted(wanted), type=pa.string())
    output = RUN / 'local_originals'
    output.mkdir(exist_ok=True)
    found, lock = {}, threading.Lock()
    entries = json.loads((RUN / 'original_catalog.json').read_text())
    def scan(entry):
        assert not (ROOT / 'REPAIR_IN_PROGRESS.json').exists()
        parquet = pq.ParquetFile(NATIVE / entry['path'])
        images = parquet.read(columns=['images.list.element.sha256'], use_threads=False)['images']
        sha = pc.struct_field(pc.list_flatten(images), 'sha256')
        if pc.any(pc.is_in(sha, value_set=hashes)).as_py():
            for row in parquet.read(columns=['images'], use_threads=False).to_pylist():
                for ref in row['images']:
                    if ref['sha256'] not in wanted:
                        continue
                    with (ROOT / entry['source_shard']).open('rb') as stream:
                        stream.seek(ref['offset_data'])
                        data = stream.read(ref['size'])
                    assert hashlib.sha256(data).hexdigest() == ref['sha256']
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        record = dict(sha256=ref['sha256'], width=image.width, height=image.height, format=image.format,
                            size=len(data), path=str(output / (ref['sha256'] + '.bin')),
                            donor_shard=entry['source_shard'], donor_member=ref['member'])
                    with lock:
                        if ref['sha256'] not in found:
                            temp = output / (ref['sha256'] + '.tmp')
                            temp.write_bytes(data)
                            os.replace(temp, record['path'])
                            found[ref['sha256']] = record
        return len(sha)
    images = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, count in enumerate(pool.map(scan, entries), 1):
            images += count
            if index % 250 == 0:
                print(index, 'shards', images, 'images', len(found), 'exact donors', flush=True)
    save(RUN / 'LOCAL_DONOR_SCAN.json', dict(wanted=len(wanted), found=len(found), donors=found,
        shards=len(entries), images=images, elapsed=time.time()-started, completed=time.time()))
    print('DONE', len(found), 'exact originals', flush=True)
    return found


def apply(found):
    changed, total = 0, 0
    for path in (RUN / 'upstream').glob('*/DONE.json'):
        directory = path.parent
        with (directory / 'lock').open('a') as lock:
            fcntl.lockf(lock, fcntl.LOCK_EX)
            rows = json.loads((directory / 'results.json').read_text())
            updates = []
            for row in rows:
                donor = found.get(row.get('original_sha256'))
                if row['status'] != 'restored' and row.get('face_bboxes') and donor:
                    assert hashlib.sha256(Path(donor['path']).read_bytes()).hexdigest() == row['original_sha256']
                    previous = {k: row.pop(k) for k in ('status', 'http_status', 'error_type', 'downloaded_sha256') if k in row}
                    row.update(donor, status='restored', recovery_source='verified_existing_baseline_bytes', previous_failure=previous)
                    updates.append(row)
                total += row.get('recovery_source') == 'verified_existing_baseline_bytes'
            if updates:
                with (directory / 'results.jsonl').open('a') as stream:
                    for row in updates:
                        stream.write(json.dumps(row) + '\n')
                save(directory / 'results.json', rows)
                done = json.loads(path.read_text())
                done['counts'] = dict(Counter(r['status'] for r in rows))
                save(path, done)
                changed += len(updates)
    save(RUN / 'LOCAL_DONORS_APPLIED.json', dict(newly_restored=changed, total_restored=total, completed=time.time()))


def retry_deferred_originals():
    import worker
    pending, deferred = [], 0
    for path in (RUN / 'upstream').glob('*/DONE.json'):
        rows = json.loads((path.parent / 'results.json').read_text())
        limited = [r for r in rows if (r['status'] in ('rate_limited', 'image_exceeds_32mib') or
                   r.get('error_type') == 'BodyDeadlineExceeded' or
                   (r['status'] == 'http_error' and r.get('http_status') == 429)) and not r.get('retry_phase')]
        if not limited:
            continue
        metadata = {r['baseline_sha256']: r for r in json.loads((path.parent / 'metadata.json').read_text())}
        for row in limited:
            if row.get('retry_not_before', 0) > time.time():
                deferred += 1
            else:
                pending.append((path.parent, metadata[row['baseline_sha256']],
                    row['status'] == 'image_exceeds_32mib' or row.get('error_type') == 'BodyDeadlineExceeded'))
    updates = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(worker.recover, row, directory, large): directory for directory, row, large in pending}
        for future in as_completed(futures):
            result = future.result()
            result['retry_phase'] = 'after_full_upstream_scan'
            updates[futures[future]][result['baseline_sha256']] = result
    for directory, changed in updates.items():
        with (directory / 'lock').open('a') as lock:
            fcntl.lockf(lock, fcntl.LOCK_EX)
            rows = json.loads((directory / 'results.json').read_text())
            rows = [changed.get(r['baseline_sha256'], r) for r in rows]
            with (directory / 'results.jsonl').open('a') as stream:
                for row in changed.values():
                    stream.write(json.dumps(row) + '\n')
            save(directory / 'results.json', rows)
            done = json.loads((directory / 'DONE.json').read_text())
            done['counts'] = dict(Counter(r['status'] for r in rows))
            save(directory / 'DONE.json', done)
    save(RUN / 'FINAL_URL_RETRY.json', dict(attempted=len(pending), deferred_until_server_allows=deferred, completed=time.time()))


if __name__ == '__main__':
    main()
