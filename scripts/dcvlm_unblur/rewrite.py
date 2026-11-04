"""Verify every member, stage a complete TAR, then atomically replace its path."""
import copy
import fcntl
import hashlib
import io
import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare import NATIVE, ROOT, RUN, save

DATASET_ID = 'local/dcvlm-baseline-200b-datacomp-url-restored@20261104'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def rewrite(entry, replacements, *, root=ROOT, native=NATIVE, run=RUN):
    relative = entry['source_shard']
    directory = run / 'rewrite' / Path(relative).stem
    directory.mkdir(parents=True, exist_ok=True)
    source = root / relative
    staged = source.with_suffix('.tar.unblur.tmp')
    with (directory / 'lock').open('w') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        if (directory / 'APPLIED.json').exists():
            return json.loads((directory / 'APPLIED.json').read_text())
        # A journal written before rename also handles a process killed after rename.
        journal = directory / 'PREPARED.json'
        if journal.exists():
            plan = json.loads(journal.read_text())
            actual = file_sha(source)
            if actual == plan['new_sha256']:
                save(directory / 'APPLIED.json', plan)
                return plan
            assert actual == plan['old_sha256'], 'Source differs from both recorded states'
            assert file_sha(staged) == plan['new_sha256']
            os.replace(staged, source)
            save(directory / 'APPLIED.json', plan)
            return plan
        assert file_sha(native / entry['path']) == entry['sha256'], 'Original index changed'
        table = pq.read_table(native / entry['path'])
        rows = table.to_pylist()
        assert len(rows) == entry['rows']
        original = json.loads(rows[0]['original_locator_json'])
        old_sha = original['archive_sha256']
        assert source.stat().st_size == original['archive_size']
        assert file_sha(source) == old_sha, 'Original TAR hash changed'
        expected, replace_members = {}, {}
        for row in rows:
            for component in json.loads(row['original_locator_json'])['components']:
                assert component['member'] not in expected
                expected[component['member']] = component
            if row['source_key'] in replacements:
                assert row['source'] == 'datacomp_1b' and len(row['images']) == 1
                ref = row['images'][0]
                restored = replacements[row['source_key']]
                assert restored['baseline_sha256'] == ref['sha256']
                assert restored['sha256'] == restored['original_sha256']
                assert restored['status'] == 'restored' and restored['face_bboxes']
                assert file_sha(restored['path']) == restored['sha256']
                replace_members[ref['member']] = restored
        assert len(replace_members) == len(replacements)
        backup = directory / 'original_images.tar'
        new_components = {}
        with tarfile.open(source, 'r:') as old, tarfile.open(staged, 'w', format=tarfile.PAX_FORMAT) as new, tarfile.open(backup, 'w') as saved:
            for member in old:
                assert member.isfile() and member.name in expected
                component = expected[member.name]
                assert member.offset_data == component['offset_data'] and member.size == component['size']
                data = old.extractfile(member).read()
                assert digest(data) == component['sha256'], member.name
                changed = copy.copy(member)
                changed.pax_headers = dict(member.pax_headers)
                if member.name in replace_members:
                    saved.addfile(member, io.BytesIO(data))
                    data = Path(replace_members[member.name]['path']).read_bytes()
                    changed.size = len(data)
                    changed.pax_headers.pop('size', None)
                new.addfile(changed, io.BytesIO(data))
        # Read the staged archive back: verify all member bytes and record actual offsets.
        with tarfile.open(staged, 'r:') as archive:
            for member in archive:
                assert member.name not in new_components
                wanted_sha = replace_members.get(member.name, {}).get('sha256', expected[member.name]['sha256'])
                assert digest(archive.extractfile(member).read()) == wanted_sha
                new_components[member.name] = dict(member=member.name, offset_data=member.offset_data,
                                                    size=member.size, sha256=wanted_sha)
        assert new_components.keys() == expected.keys()
        new_sha, new_size = file_sha(staged), staged.stat().st_size
        id_map = []
        for row in rows:
            locator = json.loads(row['original_locator_json'])
            locator.update(dataset_id=DATASET_ID, archive_sha256=new_sha, archive_size=new_size)
            locator['components'] = [new_components[c['member']] for c in locator['components']]
            before_id = row['sample_id']
            row['sample_id'] = digest(canonical(dict(dataset_id=DATASET_ID, source_shard=relative,
                                    source_key=row['source_key'], components=locator['components'])).encode())
            id_map.append(dict(old_sample_id=before_id, new_sample_id=row['sample_id'],
                               image_changed=row['source_key'] in replacements))
            for ref in row['images']:
                c = new_components[ref['member']]
                ref.update(offset=c['offset_data'], offset_data=c['offset_data'], length=c['size'],
                           size=c['size'], sha256=c['sha256'])
                if ref['member'] in replace_members:
                    restored = replace_members[ref['member']]
                    ref.update(width=restored['width'], height=restored['height'])
                meta = json.loads(ref['metadata_json'])
                meta.update({k: v for k, v in ref.items() if k != 'metadata_json'})
                meta.update(dataset_id=DATASET_ID, archive_sha256=new_sha, archive_size=new_size)
                ref['metadata_json'] = canonical(meta)
            supervision = json.loads(row['original_supervision_json'])
            supervision['sample_id'] = row['sample_id']
            adapter = json.loads(row['adapter_metadata_json'])
            adapter.update(dataset_id=DATASET_ID, previous_sample_id=before_id)
            row.update(original_locator_json=canonical(locator), original_supervision_json=canonical(supervision),
                       adapter_metadata_json=canonical(adapter))
        index = directory / 'data.parquet'
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), index, compression='zstd')
        pq.write_table(pa.Table.from_pylist(id_map), directory / 'sample_id_map.parquet', compression='zstd')
        # File content is flushed before publishing either the journal or the replacement.
        for path in (staged, backup, index, directory / 'sample_id_map.parquet'):
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        assert file_sha(source) == old_sha, 'Source was modified while preparing the replacement'
        plan = dict(source_shard=relative, old_sha256=old_sha, new_sha256=new_sha, new_size=new_size,
                    changed_images=len(replace_members), records=len(rows), index=str(index),
                    index_sha256=file_sha(index), dataset_id=DATASET_ID, prepared=time.time(),
                    replacements=replacements, backup=str(backup),
                    source_counts=dict(Counter(r['source'] for r in rows)),
                    format_counts=dict(Counter(r['format'] for r in rows)),
                    image_slots=sum(r['image_count'] for r in rows))
        save(journal, plan)
        os.replace(staged, source)
        directory_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        save(directory / 'APPLIED.json', plan)
        return plan
