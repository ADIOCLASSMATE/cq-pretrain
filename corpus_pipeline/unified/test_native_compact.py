"""Lossless native metadata compaction, provenance and resumability tests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.production.common import atomic_json, file_sha
from . import native_compact as compact
from .ingest import arrow_schema


def row(source, shard, key):
    # Deliberately retain punctuation, unicode, nulls, nested ordering and logical image paths.
    return dict(collection='dcvlm', source=source, source_shard=shard, source_key=key, sample_id='id-' + key,
        format='instruction', segments=[dict(kind='text', text='  原始文本\n\t"<>"  ', role='human', turn=0,
        image_index=None, source_member=None), dict(kind='image', text=None, role='human', turn=0,
        image_index=0, source_member=key + '.0.jpg')],
        images=[dict(kind='tar', path='hf://repo@revision/' + shard, member=key + '.0.jpg', offset_data=512,
        size=99, offset=512, length=99, sha256='a'*64, width=None, height=None,
        metadata_json='{"dataset_id":"repo@revision","relative_path":"'+shard+'"}')],
        image_count=1, text_hash='b'*64, original_supervision_json=' {"literal": "\\n", "roles": ["human"]} ',
        original_locator_json='{"adapter":"portable","key":"'+key+'"}',
        adapter_metadata_json='{"keep_order":true}')


def fixture(root):
    index = root / 'index'; index.mkdir()
    data = [('source-a', 'shards/000/000000.tar', ['a', 'b']),
            ('source-b', 'shards/000/000000.tar', ['c']),
            ('source-a', 'shards/000/000001.tar', ['d']),
            ('source-b', 'shards/000/000001.tar', ['e', 'f'])]
    catalog = []
    for number, (source, shard, keys) in enumerate(data):
        path = index / f'part-{number}.parquet'
        pq.write_table(pa.Table.from_pylist([row(source, shard, key) for key in keys], schema=arrow_schema()), path)
        catalog.append(dict(path=path.name, bytes=path.stat().st_size, sha256=file_sha(path), rows=len(keys),
                            collection='dcvlm', source=source, source_shard=shard))
    plan = dict(repo='mlfoundations/dcvlm-baseline-200b', revision='test-pinned', source_total_files=2,
                source_total_bytes=1234, source_inventory_sha256='c'*64)
    atomic_json(index / 'PLAN.json', plan)
    atomic_json(index / 'catalog.json', catalog)
    manifest = dict(schema_version='corpus.native.v2', phase='complete', scope='full_pinned_upstream_release',
        collections=['dcvlm'], statistics=dict(samples=6, image_slots=6, sources={'source-a': 3, 'source-b': 3}),
        plan_sha256=file_sha(index/'PLAN.json'), portable_datasets=['repo@revision'], allowed_image_roots=[],
        source_inventory_sha256=plan['source_inventory_sha256'],
        source_validation=dict(repo=plan['repo'], revision=plan['revision'], files=2, bytes=1234, complete=True))
    atomic_json(index / 'manifest.json', manifest)
    atomic_json(index / 'COMPLETE.json', dict(schema_version='corpus.native.v2', complete=True, validation_passed=True,
        scope=manifest['scope'], statistics=manifest['statistics'], plan_sha256=file_sha(index/'PLAN.json'),
        manifest_sha256=file_sha(index/'manifest.json'), catalog_sha256=file_sha(index/'catalog.json')))
    return index


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = fixture(self.root)
        self.output = self.root / 'native'
        self.space = patch.object(compact.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(4<<40, 1<<40, 3<<40))
        self.space.start(); self.addCleanup(self.space.stop)

    def test_direct_arrow_rows_unchanged_and_one_file_per_tar(self):
        before = {p.name: (file_sha(p), p.stat().st_mtime_ns) for p in self.index.iterdir()}
        result = compact.run(self.index, self.output, workers=2, batch_rows=1)
        self.assertTrue(result['validation_passed'])
        self.assertEqual(result['scope'], 'full_declared_inputs')
        catalog = compact.read(self.output/'catalog.json')
        self.assertEqual(len(catalog), 2)
        input_rows, output_rows = {}, {}
        for entry in compact.read(self.index/'catalog.json'):
            for value in pq.read_table(self.index/entry['path']).to_pylist():
                input_rows[value['sample_id']] = value
        for entry in catalog:
            self.assertEqual(entry['source'], '__mixed__')
            self.assertEqual(entry['sources'], ['source-a', 'source-b'])
            self.assertEqual(entry['contract'], 'mixed_source_tar.v1')
            for value in pq.read_table(self.output/entry['path']).to_pylist():
                self.assertEqual(value['source_shard'], entry['source_shard'])
                output_rows[value['sample_id']] = value
        self.assertEqual(input_rows, output_rows)
        self.assertEqual(before, {p.name: (file_sha(p), p.stat().st_mtime_ns) for p in self.index.iterdir()})
        manifest = compact.read(self.output/'manifest.json')
        self.assertEqual(manifest['catalog_contract'], 'mixed_source_tar.v1')
        self.assertEqual(manifest['enumeration_scope'], 'derived_full_pinned_source_metadata')
        self.assertEqual(manifest['source_lineage']['input_files_covered_once'], 4)
        for name in ('PLAN.json', 'COMPLETE.json', 'manifest.json', 'catalog.json'):
            self.assertEqual(file_sha(self.index/name), file_sha(self.output/'lineage'/name))
        self.assertEqual(compact.finalize(self.index, self.output), result)

    def test_bounded_groups_and_resume_reuse_committed_output(self):
        compact.prepare(self.index, self.output, max_group_rows=2, batch_rows=1)
        result = compact.worker(self.index, self.output, workers=2, max_groups=1)
        self.assertEqual(result['counts']['completed'], 1)
        first = next((self.output/'groups').glob('*/data.parquet'))
        before = (file_sha(first), first.stat().st_mtime_ns)
        with self.assertRaises(compact.PendingGroups):
            compact.finalize(self.index, self.output)
        result = compact.worker(self.index, self.output, workers=2)
        self.assertEqual(result['counts']['reused'], 1)
        self.assertEqual(before, (file_sha(first), first.stat().st_mtime_ns))
        self.assertEqual(compact.finalize(self.index, self.output)['statistics']['samples'], 6)

    def test_input_checksum_mutation_fails_without_complete(self):
        compact.prepare(self.index, self.output)
        with (self.index/'part-0.parquet').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            compact.worker(self.index, self.output, workers=1)
        self.assertFalse((self.output/'COMPLETE.json').exists())

    def test_partial_snapshot_or_stale_metadata_cannot_be_promoted(self):
        receipt = compact.read(self.index/'COMPLETE.json')
        receipt['scope'] = 'validation'
        atomic_json(self.index/'COMPLETE.json', receipt)
        with self.assertRaisesRegex(ValueError, 'fully verified'):
            compact.prepare(self.index, self.output)
        receipt['scope'] = 'full_pinned_upstream_release'
        receipt['catalog_sha256'] = '0'*64
        atomic_json(self.index/'COMPLETE.json', receipt)
        with self.assertRaisesRegex(ValueError, 'binding'):
            compact.prepare(self.index, self.output)

    def test_plan_cannot_omit_or_duplicate_an_input(self):
        compact.prepare(self.index, self.output)
        plan = compact.read(self.output/'PLAN.json')
        plan['groups'][0]['inputs'].pop()
        atomic_json(self.output/'PLAN.json', plan)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            compact.worker(self.index, self.output)

    def test_row_source_must_equal_input_catalog(self):
        path = self.index/'part-0.parquet'
        pq.write_table(pa.Table.from_pylist([row('foreign', 'shards/000/000000.tar', 'a'),
                                           row('foreign', 'shards/000/000000.tar', 'b')], schema=arrow_schema()), path)
        catalog = compact.read(self.index/'catalog.json')
        catalog[0].update(bytes=path.stat().st_size, sha256=file_sha(path))
        atomic_json(self.index/'catalog.json', catalog)
        complete = compact.read(self.index/'COMPLETE.json'); complete['catalog_sha256'] = file_sha(self.index/'catalog.json')
        atomic_json(self.index/'COMPLETE.json', complete)
        compact.prepare(self.index, self.output)
        with self.assertRaisesRegex(ValueError, 'membership'):
            compact.worker(self.index, self.output, workers=1)

    def test_output_corruption_blocks_resume_and_finalize(self):
        compact.prepare(self.index, self.output)
        compact.worker(self.index, self.output, workers=2)
        path = next((self.output/'groups').glob('*/data.parquet'))
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            compact.finalize(self.index, self.output)
        self.assertFalse((self.output/'COMPLETE.json').exists())

    def test_space_reserve_cannot_be_lowered_or_ignored(self):
        compact.prepare(self.index, self.output)
        with self.assertRaisesRegex(ValueError, '1 TiB'):
            compact.worker(self.index, self.output, reserve_bytes=0)
        with patch.object(compact.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(4<<40, 4<<40, 0)):
            with self.assertRaises(compact.PendingGroups):
                compact.worker(self.index, self.output, workers=1)
        self.assertFalse((self.output/'COMPLETE.json').exists())

    def test_thread_and_process_locks_exclude_duplicate_group_writers(self):
        lock = self.root/'group.lock'
        with compact.claim(lock) as held:
            self.assertTrue(held)
            with ThreadPoolExecutor(1) as pool:
                def probe():
                    with compact.claim(lock) as other:
                        return other
                self.assertFalse(pool.submit(probe).result())
            code = "import fcntl,sys; f=open(sys.argv[1],'a+');\ntry: fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(9)\n"
            self.assertEqual(subprocess.run([sys.executable, '-c', code, str(lock)]).returncode, 9)

    def test_uncommitted_partial_is_rebuilt_and_inputs_stay_readonly(self):
        compact.prepare(self.index, self.output)
        group = compact.read(self.output/'PLAN.json')['groups'][0]
        directory = self.output/'groups'/group['group_id']; directory.mkdir(parents=True)
        (directory/'data.parquet.partial').write_text('interrupted writer')
        (directory/'data.parquet').write_text('uncommitted previous output')
        compact.worker(self.index, self.output, workers=2)
        self.assertTrue(compact.finalize(self.index, self.output)['complete'])

    def test_complete_snapshot_passes_existing_full_release_gate(self):
        from corpus_pipeline.production.records import snapshot
        from corpus_pipeline.unified.release import _verify_full_native_snapshot
        from corpus_pipeline.production.common import digest
        compact.run(self.index, self.output, workers=2)
        snap = snapshot(self.output)
        catalog = compact.read(self.output/'catalog.json')
        self.assertEqual(len(snap['catalog_index']), 2)
        artifacts = [{'path': str(self.output/name), 'sha256': file_sha(self.output/name)}
                     for name in ('COMPLETE.json', 'manifest.json', 'catalog.json')]
        collections = {'dcvlm': digest({'snapshot': file_sha(self.output/'COMPLETE.json'),
            'collection': 'dcvlm', 'files': catalog})}
        _verify_full_native_snapshot({'artifacts': artifacts, 'coverage': {'processed': 6}}, collections)

    def test_partial_symlink_cannot_overwrite_source_metadata(self):
        compact.prepare(self.index, self.output)
        group = compact.read(self.output/'PLAN.json')['groups'][0]
        directory = self.output/'groups'/group['group_id']; directory.mkdir(parents=True)
        source = self.index/'part-0.parquet'; before = file_sha(source)
        (directory/'data.parquet.partial').symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'Unsafe compaction'):
            compact.worker(self.index, self.output, workers=1)
        self.assertEqual(file_sha(source), before)

    def test_input_output_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            compact.prepare(self.index, self.index/'native')


if __name__ == '__main__':
    unittest.main()
