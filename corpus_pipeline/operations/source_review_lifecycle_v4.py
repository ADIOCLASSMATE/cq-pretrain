"""Run the frozen review farm and strict export; never mutate original datasets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run_parameters(binding):
    values = {key: binding.get(key, default) for key, default in
              [('initial_concurrency', 4), ('ceiling', 64), ('batch_size', 4)]}
    if any(type(value) is not int for value in values.values()):
        raise ValueError('Lifecycle concurrency and batch parameters must be integers')
    if not 1 <= values['initial_concurrency'] <= values['ceiling'] <= 64:
        raise ValueError('Lifecycle requires 1 <= initial_concurrency <= ceiling <= 64')
    if not 1 <= values['batch_size'] <= 8:
        raise ValueError('Lifecycle requires 1 <= batch_size <= 8')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    args = parser.parse_args()
    binding_path = Path(args.binding).resolve(strict=True)
    binding = json.loads(binding_path.read_text())
    parameters = run_parameters(binding)
    base = binding_path.parent
    with (base / 'lifecycle.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in binding['required_files']:
            if sha(item['path']) != item['sha256']:
                raise RuntimeError('Lifecycle binding changed: ' + item['path'])
        acceptance = json.loads(Path(binding['acceptance']).read_text())
        if acceptance.get('approved_for_full_review') is not True:
            raise RuntimeError('Full review lacks successful actual-image canary acceptance')
        status_path = base / 'LIFECYCLE_STATUS.json'
        status = {'phase': 'review_running', 'pid': os.getpid(), 'started_at': time.time(),
                  'binding_sha256': sha(binding_path), 'full_review_complete': False,
                  'adoption_complete': False, 'original_datasets_modified': False, 'run_parameters': parameters}
        save(status_path, status)
        farm = Path(binding['farm_root'])
        command = [sys.executable, binding['farm_code'], 'run', '--root', str(farm),
                   '--initial-concurrency', str(parameters['initial_concurrency']), '--ceiling', str(parameters['ceiling']), '--batch-size', str(parameters['batch_size']),
                   '--codex-binary', '/root/.bun/bin/codex']
        with (base / 'farm-run.log').open('ab') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        db = sqlite3.connect(f'file:{farm}/state.sqlite3?mode=ro', uri=True)
        counts = dict(db.execute('SELECT status,count(*) FROM items GROUP BY status'))
        db.close()
        status.update(review_exitcode=completed.returncode, counts=counts, updated_at=time.time())
        if completed.returncode or counts != {'succeeded': binding['expected_items']}:
            status.update(phase='review_needs_attention', reason='Incomplete review; no strict completion or adoption claimed.')
            save(status_path, status)
            raise SystemExit(2)
        status['phase'] = 'strict_audit_and_export'
        save(status_path, status)
        with (base / 'farm-export.log').open('ab') as log:
            exported = subprocess.run([sys.executable, binding['farm_code'], 'export',
                                       '--root', str(farm), '--shard-size', '1000'],
                                      stdout=log, stderr=subprocess.STDOUT)
        if exported.returncode:
            status.update(phase='audit_needs_attention', export_exitcode=exported.returncode,
                          updated_at=time.time())
            save(status_path, status)
            raise SystemExit(3)
        final = farm / 'FINAL.json'
        result = json.loads(final.read_text())
        if result.get('complete') is not True or result.get('records') != binding['expected_items']:
            raise RuntimeError('Unexpected strict FINAL coverage')
        status.update(phase='review_strict_complete_awaiting_revision_adoption',
                      full_review_complete=True, final={'path': str(final), 'sha256': sha(final)},
                      updated_at=time.time())
        save(status_path, status)


if __name__ == '__main__':
    main()
