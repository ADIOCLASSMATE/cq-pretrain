"""Exercise replacement, unchanged members, rejection and post-rename recovery."""
import copy
import io
import json
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare import NATIVE, ROOT, RUN, save
from rewrite import canonical, digest, file_sha, rewrite


def main():
    catalog = json.loads((RUN / 'original_catalog.json').read_text())
    entry = next(x for x in catalog if x['source_shard'] == 'shards/000/000003.tar')
    table = pq.read_table(NATIVE / entry['path'])
    row = next(x for x in table.to_pylist() if x['source_key'] == 'r00247_000170180')
    report = Path(__file__).resolve().parents[2] / 'reports/dcvlm_blur_audit_20261104'
    restored = report / 'r00247_000170180_redownloaded.jpeg'
    meta = json.loads((report / 'metadata_r00247_000170180.json').read_text())['metadata']
    replacement = dict(status='restored', baseline_sha256=row['images'][0]['sha256'],
        original_sha256=meta['sha256'], sha256=file_sha(restored), face_bboxes=meta['face_bboxes'],
        path=str(restored), width=300, height=450)
    with tempfile.TemporaryDirectory(prefix='rewrite_test_', dir=RUN) as folder:
        root = Path(folder) / 'data'
        native = Path(folder) / 'native'
        run = Path(folder) / 'run'
        root.mkdir(); native.mkdir(); run.mkdir()
        payloads = {}
        with (ROOT / entry['source_shard']).open('rb') as original:
            for c in json.loads(row['original_locator_json'])['components']:
                original.seek(c['offset_data'])
                payloads[c['member']] = original.read(c['size'])
        rows = [copy.deepcopy(row), copy.deepcopy(row)]
        rows[1]['source_key'] = 'unchanged'
        rows[1]['sample_id'] = 'unchanged_id'
        rows[1]['source'] = 'control_source'
        rows[1]['adapter_metadata_json'] = canonical(dict(json.loads(rows[1]['adapter_metadata_json']), prepared_adapter_status='quarantine'))
        archive = root / 'fixture.tar'
        with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT) as target:
            for r in rows:
                for name, data in payloads.items():
                    renamed = name.replace(row['source_key'], r['source_key'])
                    info = tarfile.TarInfo(renamed); info.size = len(data)
                    target.addfile(info, io.BytesIO(data))
        with tarfile.open(archive) as source:
            members = {m.name: dict(member=m.name, offset_data=m.offset_data, size=m.size,
                                   sha256=digest(source.extractfile(m).read())) for m in source}
        for r in rows:
            r['source_shard'] = 'fixture.tar'
            locator = json.loads(r['original_locator_json'])
            locator.update(archive_sha256=file_sha(archive), archive_size=archive.stat().st_size, path='fixture.tar')
            locator['components'] = [members[c['member'].replace(row['source_key'], r['source_key'])] for c in locator['components']]
            r['original_locator_json'] = canonical(locator)
            for ref in r['images']:
                name = ref['member'].replace(row['source_key'], r['source_key'])
                c = members[name]
                ref.update(member=name, path='fixture.tar', offset=c['offset_data'], offset_data=c['offset_data'])
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), native / 'data.parquet')
        fixture = dict(source_shard='fixture.tar', path='data.parquet', rows=2,
                       sha256=file_sha(native / 'data.parquet'))
        original_sha = file_sha(archive)
        unchanged_run = Path(folder) / 'unchanged_run'
        unchanged_run.mkdir()
        before_stat = archive.stat()
        unchanged = rewrite(fixture, {}, root=root, native=native, run=unchanged_run)
        assert unchanged['unchanged'] and unchanged['new_sha256'] == original_sha
        assert unchanged['index'] == str(native / 'data.parquet')
        assert archive.stat().st_ino == before_stat.st_ino and archive.stat().st_mtime_ns == before_stat.st_mtime_ns
        bad = dict(replacement, sha256='0' * 64)
        try:
            rewrite(fixture, {row['source_key']: bad}, root=root, native=native, run=run)
        except AssertionError:
            pass
        else:
            raise AssertionError('A mismatching replacement was accepted')
        assert file_sha(archive) == original_sha
        result = rewrite(fixture, {row['source_key']: replacement}, root=root, native=native, run=run)
        assert result['changed_images'] == 1 and file_sha(archive) == result['new_sha256']
        updated = pq.read_table(result['index']).to_pylist()
        assert updated[0]['segments'] == rows[0]['segments']
        assert updated[1]['images'][0]['sha256'] == rows[1]['images'][0]['sha256']
        assert updated[0]['sample_id'] != rows[0]['sample_id']
        with archive.open('rb') as source:
            for r in updated:
                for c in json.loads(r['original_locator_json'])['components']:
                    source.seek(c['offset_data'])
                    assert digest(source.read(c['size'])) == c['sha256']
        with tarfile.open(result['backup']) as backup:
            assert backup.extractfile(row['images'][0]['member']).read() == payloads[row['images'][0]['member']]
        (run / 'rewrite/fixture/APPLIED.json').unlink()
        resumed = rewrite(fixture, {row['source_key']: replacement}, root=root, native=native, run=run)
        assert resumed == result
        assert rewrite(fixture, {row['source_key']: replacement}, root=root, native=native, run=run) == result
        assert file_sha(archive) == result['new_sha256']
        import pipeline
        pipeline.RUN, pipeline.ROOT = run, root
        fixture.update(collection='dcvlm', source='__mixed__', contract='mixed_source_tar.v1',
                       sources=['datacomp_1b', 'control_source'])
        save(run / 'original_catalog.json', [fixture])
        save(run / 'COVERAGE.json', dict(record_counts={'restored': 1, 'no_blur_annotation': 1},
            unresolved_records=0, failure_policy='test', samples_deleted=0))
        save(run / 'APPLY_READY.json', {'jobs': ['fixture.json']})
        final = pipeline.finalize()
        assert final['restored_records'] == 1 and final['all_datacomp_blur_removed']
        complete = json.loads((run / 'native/COMPLETE.json').read_text())
        assert complete['statistics']['samples'] == 2 and complete['statistics']['image_slots'] == 2
        assert complete['statistics']['adapter_statuses'] == {'ready': 1, 'quarantine': 1}
    print('PASS: rejection, exact restored bytes, unchanged members, all locator offsets, backup, restart, idempotence and native reader publication')


if __name__ == '__main__':
    main()
