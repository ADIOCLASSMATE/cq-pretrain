"""Pinned queue, immutable source, portable identities, and publication gates."""
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from corpus_pipeline.production.common import atomic_json, canonical, file_sha
from corpus_pipeline.production.records import snapshot
from corpus_pipeline.production.records import load_raw_record
from corpus_pipeline.production.ingest import build_shard_manifest, iter_records
from corpus_pipeline.cluster.frontpass import record_decision
from corpus_pipeline.production.portable_refs import read_ref_bytes
from . import baseline_index as index
from .ingest import _balanced_rows
from .test_ingest import balanced
from .test_baseline_compat import EXAMPLES


def fixture(root, count=2):
    raw, control = root / 'raw', root / 'download'
    control.mkdir(parents=True)
    files, siblings, payloads = [], [], {}
    for ordinal in range(count):
        relative = 'shards/000/%06d.tar' % ordinal
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Byte ranges are indexed without decoding images or changing text.
        image = b'original-image-payload-' + str(ordinal).encode()
        payloads[relative] = image
        members = [('a.conversations.txt', b'human:Keep spaces:  <image>\nQuestion?<EOCL>\ngpt:Exact answer.\n'),
                   ('a.data_type.txt', b'instruction'), ('a.ds_name.txt', b'visual-source'),
                   ('a.orig_url.txt', b'https://example.test/original'), ('a.ntok.txt', b'19'),
                   ('a.mdp.txt', b'2'), ('a.0.png', image),
                   ('b.conversations.txt', b'human:Text only?<EOCL>\ngpt:Yes.'),
                   ('b.data_type.txt', b'text'), ('b.ds_name.txt', b'text-source'),
                   ('b.orig_url.txt', b'https://example.test/text')]
        with tarfile.open(path, 'w') as archive:
            for name, data in members:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        sha = file_sha(path)
        files.append(dict(id='%05d' % ordinal, path=relative, size=path.stat().st_size,
                          sha256=sha, git_blob_oid='a' * 40))
        siblings.append(dict(rfilename=relative, size=path.stat().st_size,
                             blobId='a' * 40, lfs=dict(sha256=sha)))
    readme = raw / 'README.md'
    readme.write_bytes(b'pinned release documentation')
    git = hashlib.sha1(('blob %d\0' % readme.stat().st_size).encode() + readme.read_bytes()).hexdigest()
    files.append(dict(id='%05d' % count, path='README.md', size=readme.stat().st_size,
                      sha256=None, git_blob_oid=git))
    siblings.append(dict(rfilename='README.md', size=readme.stat().st_size, blobId=git))
    atomic_json(control / 'dataset_metadata.json', dict(id=index.REPO, sha=index.REVISION, siblings=siblings))
    manifest = dict(repo=index.REPO, revision=index.REVISION, destination=str(raw), files=files,
                    total_files=len(files), total_bytes=sum(row['size'] for row in files),
                    metadata_sha256=file_sha(control / 'dataset_metadata.json'))
    atomic_json(control / 'manifest.json', manifest)
    return raw, control, manifest, payloads


def mark(control, manifest, relative):
    entry = next(row for row in manifest['files'] if row['path'] == relative)
    path = Path(manifest['destination']) / relative
    stat = path.stat()
    atomic_json(control / 'verified' / (entry['id'] + '.json'), dict(
        revision=index.REVISION, path=relative, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        source_sha256=entry['sha256'], source_git_blob_oid=entry['git_blob_oid'], sha256=file_sha(path)))


def mark_all(control, manifest):
    for entry in manifest['files']:
        mark(control, manifest, entry['path'])


def source_complete(control, manifest):
    atomic_json(control / 'COMPLETE.json', dict(revision=index.REVISION,
        files=manifest['total_files'], bytes=manifest['total_bytes']))


def lock_probe(path, queue):
    with index._lock(Path(path)) as acquired:
        queue.put(acquired)


class BaselineIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw, self.control, self.manifest, self.payloads = fixture(self.root)
        self.output = self.root / 'preprocessing/native'
        self.count_patch = patch.object(index, 'EXPECTED_TRAINING_FILES', 2)
        self.count_patch.start()
        self.space_patch = patch.object(index, '_space')
        self.space_patch.start()

    def tearDown(self):
        self.space_patch.stop()
        self.count_patch.stop()
        self.temp.cleanup()

    def prepare(self, **kwargs):
        return index.prepare(self.control, self.output, **kwargs)

    def test_quarantine_survives_portable_reader_and_frontpass(self):
        balanced(self.raw, EXAMPLES)
        metadata = json.loads((self.control/'dataset_metadata.json').read_text())
        relative = 'shards/000/000000.tar'
        path = self.raw / relative
        entry = next(item for item in self.manifest['files'] if item['path'] == relative)
        entry.update(size=path.stat().st_size, sha256=file_sha(path))
        sibling = next(item for item in metadata['siblings'] if item['rfilename'] == relative)
        sibling.update(size=entry['size'], lfs={'sha256': entry['sha256']})
        atomic_json(self.control/'dataset_metadata.json', metadata)
        self.manifest.update(total_bytes=sum(item['size'] for item in self.manifest['files']),
                             metadata_sha256=file_sha(self.control/'dataset_metadata.json'))
        atomic_json(self.control/'manifest.json', self.manifest)
        mark_all(self.control, self.manifest)
        plan = self.prepare()
        index.worker(self.control, self.output, 'a')
        source_complete(self.control, self.manifest)
        result = index.finalize(self.control, self.output)
        self.assertEqual(result['statistics']['adapter_statuses'], {'ready': 2, 'quarantine': 2})
        roots = self.root/'roots.json'
        atomic_json(roots, {plan['dataset_id']: str(self.raw)})
        with patch.dict(os.environ, {'CORPUS_DATASET_ROOTS': str(roots)}):
            loaded = list(iter_records(build_shard_manifest(self.output), snapshot(self.output), {}))
        quarantined = [row for row in loaded if row['prepared_adapter_status'] == 'quarantine']
        self.assertEqual(len(quarantined), 2)
        for row in quarantined:
            self.assertEqual(row['normalized']['decision'], 'quarantine')
            self.assertIn('unresolved_original_image_binding', row['normalized']['reason_codes'])
            self.assertIn('unreferenced_images', row['normalized']['issues'])
            self.assertTrue(all(ref['dataset_id'] == plan['dataset_id'] for ref in row['source_image_refs']))
            decision, reasons = record_decision(row, ['no_detector_alert'] * len(row['source_image_refs']))
            self.assertEqual(decision, 'pending')
            self.assertIn('unresolved_original_image_binding', reasons)
            broken = deepcopy(row['raw'])
            meta = json.loads(broken['adapter_metadata_json'])
            meta['prepared_adapter_status'] = 'ready'
            broken['adapter_metadata_json'] = canonical(meta)
            with self.assertRaisesRegex(ValueError, 'invalid unresolved'):
                load_raw_record(broken, {'parquet_path': row['locator']['path'],
                                        'parquet_row_index': row['locator']['row_index']},
                                snapshot(self.output), {}, dataset_roots={plan['dataset_id']: str(self.raw)})

    def test_legacy_migration_hardlinks_preserve_original_receipts_and_checksums(self):
        mark_all(self.control, self.manifest)
        legacy_output = self.root/'legacy'
        with patch.object(index, '_code_identity', return_value=index.LEGACY_CODE_IDENTITY):
            old_plan = index.prepare(self.control, legacy_output)
            index.worker(self.control, legacy_output, 'old')
        first = old_plan['units'][0]
        old_receipt = legacy_output/'units'/first['unit_id']/'COMPLETE.json'
        before = old_receipt.read_bytes()
        plan = self.prepare(reuse_from=legacy_output)
        self.assertNotIn(str(legacy_output), canonical(plan))
        self.assertEqual(self.prepare(), plan)
        result = index.migrate(self.control, self.output, legacy_output, max_units=1)
        self.assertEqual(result['units_migrated'], 1)
        migrated = self.output/'units'/first['unit_id']/'COMPLETE.json'
        self.assertEqual(migrated.read_bytes(), before)
        self.assertEqual(migrated.stat().st_ino, old_receipt.stat().st_ino)
        receipt = json.loads(before)
        self.assertEqual(receipt['code_identity'], index.LEGACY_CODE_IDENTITY)
        for item in receipt['catalog']:
            self.assertEqual((legacy_output/item['path']).stat().st_ino,
                             (self.output/item['path']).stat().st_ino)
        index.migrate(self.control, self.output, legacy_output)
        source_complete(self.control, self.manifest)
        final = index.finalize(self.control, self.output)
        self.assertEqual(final['statistics']['samples'], 4)
        self.assertEqual(final['statistics']['adapter_statuses'], {'ready': 4})
        published = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(len(published['legacy_unit_migrations']), 2)
        self.assertEqual(published['legacy_plan']['sha256'], file_sha(legacy_output/'PLAN.json'))
        migration = migrated.parent/'MIGRATION.json'
        value = json.loads(migration.read_text()); value['original_receipt_sha256'] = '0'*64
        atomic_json(migration, value)
        with self.assertRaisesRegex(ValueError, 'receipt identity mismatch'):
            index.status(self.control, self.output)

    def test_v4_migration_preserves_v3_and_v2_ancestry_and_rejects_tamper(self):
        mark_all(self.control, self.manifest)
        v2, v3 = self.root/'v2', self.root/'v3'
        with patch.object(index, '_code_identity', return_value=index.LEGACY_CODE_IDENTITY):
            old = index.prepare(self.control, v2)
            index.worker(self.control, v2, 'old', max_units=1)
        with patch.object(index, '_code_identity', return_value=index.V3_CODE_IDENTITY):
            index.prepare(self.control, v3, reuse_from=v2)
            index.migrate(self.control, v3, v2)
            index.worker(self.control, v3, 'v3')
        plan = self.prepare(reuse_from=v3)
        result = index.migrate(self.control, self.output, v3, workers=2)
        self.assertEqual(result['units_migrated'], 2)
        identities = set()
        for unit in old['units']:
            relative = Path('units')/unit['unit_id']/'COMPLETE.json'
            self.assertEqual((self.output/relative).read_bytes(), (v3/relative).read_bytes())
            self.assertEqual((self.output/relative).stat().st_ino, (v3/relative).stat().st_ino)
            receipt = json.loads((self.output/relative).read_text())
            identities.add(canonical(receipt['code_identity']))
            for entry in receipt['catalog']:
                self.assertEqual((self.output/entry['path']).stat().st_ino,
                                 (v3/entry['path']).stat().st_ino)
        self.assertEqual(identities, {canonical(index.LEGACY_CODE_IDENTITY), canonical(index.V3_CODE_IDENTITY)})
        source_complete(self.control, self.manifest)
        index.finalize(self.control, self.output)
        published = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual([item['sha256'] for item in published['legacy_plan_ancestry']],
                         [file_sha(v3/'PLAN.json'), file_sha(v2/'PLAN.json')])
        first = old['units'][0]
        migration = json.loads((self.output/'units'/first['unit_id']/'MIGRATION.json').read_text())
        parent = self.output/migration['parent_migration']['path']
        self.assertEqual(parent.read_bytes(), (v3/'units'/first['unit_id']/'MIGRATION.json').read_bytes())
        self.assertNotIn(str(self.root), canonical(plan) + canonical(migration))
        changed = json.loads(parent.read_text())
        changed['original_receipt_sha256'] = '0'*64
        atomic_json(parent, changed)  # Replace derived proof; never mutate its hardlinked origin.
        with self.assertRaisesRegex(ValueError, 'receipt identity mismatch'):
            index.status(self.control, self.output)
        with self.assertRaises(ValueError):
            index.finalize(self.control, self.output)

    def test_migration_rejects_missing_or_changed_ancestor_plan(self):
        mark_all(self.control, self.manifest)
        v2, v3 = self.root/'v2', self.root/'v3'
        with patch.object(index, '_code_identity', return_value=index.LEGACY_CODE_IDENTITY):
            index.prepare(self.control, v2)
            index.worker(self.control, v2, 'old')
        with patch.object(index, '_code_identity', return_value=index.V3_CODE_IDENTITY):
            index.prepare(self.control, v3, reuse_from=v2)
            index.migrate(self.control, v3, v2)
        proof = v3/'provenance'/('plan-' + file_sha(v2/'PLAN.json') + '.json')
        bad = json.loads(proof.read_text()); bad['code_identity'] = index.V3_CODE_IDENTITY
        atomic_json(proof, bad)
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            self.prepare(reuse_from=v3)

    def test_migration_rejects_corrupted_old_parquet_without_committing(self):
        mark_all(self.control, self.manifest)
        legacy_output = self.root/'legacy'
        with patch.object(index, '_code_identity', return_value=index.LEGACY_CODE_IDENTITY):
            old_plan = index.prepare(self.control, legacy_output)
            index.worker(self.control, legacy_output, 'old')
        first = old_plan['units'][0]
        receipt = json.loads((legacy_output/'units'/first['unit_id']/'COMPLETE.json').read_text())
        path = legacy_output/receipt['catalog'][0]['path']
        value = bytearray(path.read_bytes()); value[20] ^= 1; path.write_bytes(value)
        self.prepare(reuse_from=legacy_output)
        with self.assertRaisesRegex(ValueError, 'Parquet changed'):
            index.migrate(self.control, self.output, legacy_output)
        self.assertFalse((self.output/'units'/first['unit_id']/'COMPLETE.json').exists())

    def test_fixed_queue_waits_for_verified_final_and_complete_source(self):
        plan = self.prepare()
        self.assertEqual(len(plan['units']), 2)
        extra = self.raw / 'shards/000/999999.tar'
        extra.write_bytes(b'unlisted archive is never processed')
        extra.with_suffix('.tar.download-partial').write_bytes(b'inflight')
        initial = index.worker(self.control, self.output, 'a')
        self.assertEqual(initial['phase'], 'waiting_for_download')
        self.assertEqual(initial['units_complete'], 0)
        relative = 'shards/000/000000.tar'
        before = (file_sha(self.raw / relative), (self.raw / relative).stat().st_mtime_ns)
        mark(self.control, self.manifest, relative)
        first = index.worker(self.control, self.output, 'a')
        self.assertEqual(first['units_complete'], 1)
        self.assertEqual(first['phase'], 'waiting_for_download')
        self.assertEqual(before, (file_sha(self.raw / relative), (self.raw / relative).stat().st_mtime_ns))
        with self.assertRaises(index.PendingInputs):
            index.finalize(self.control, self.output)
        self.assertFalse((self.output / 'manifest.json').exists())
        self.assertFalse((self.output / 'COMPLETE.json').exists())
        mark_all(self.control, self.manifest)
        second = index.worker(self.control, self.output, 'a')
        self.assertEqual(second['units_complete'], 2)
        self.assertFalse(second['source_complete'])
        with self.assertRaises(index.PendingInputs):
            index.finalize(self.control, self.output)
        source_complete(self.control, self.manifest)
        self.assertEqual(index.status(self.control, self.output)['phase'], 'ready_to_finalize')
        self.assertTrue(index.finalize(self.control, self.output)['complete'])
        self.assertEqual(index.status(self.control, self.output)['phase'], 'complete')
        self.assertEqual(snapshot(self.output)['manifest']['statistics']['samples'], 4)

    def test_portable_rows_preserve_supervision_and_survive_root_and_mtime_change(self):
        plan = self.prepare(rows_per_file=1)
        mark_all(self.control, self.manifest)
        index.worker(self.control, self.output, 'a')
        source_complete(self.control, self.manifest)
        index.finalize(self.control, self.output)
        catalog = json.loads((self.output / 'catalog.json').read_text())
        rows = [row for entry in catalog for row in pq.read_table(self.output / entry['path']).to_pylist()]
        original = {}
        for unit in plan['units']:
            for row in _balanced_rows(dict(path=str(self.raw / unit['relative_path']), relative_path=unit['relative_path'], source=None)):
                original[(row['source_shard'], row['source_key'])] = row
        for row in rows:
            before = original[(row['source_shard'], row['source_key'])]
            self.assertEqual(row['segments'], before['segments'])
            expected, observed = map(json.loads, [before['original_supervision_json'], row['original_supervision_json']])
            expected['sample_id'] = observed['sample_id']
            self.assertEqual(expected, observed)
            self.assertNotIn(str(self.raw), canonical(row))
            locator = json.loads(row['original_locator_json'])
            self.assertNotIn('archive_fingerprint', locator)
            self.assertEqual(locator['dataset_id'], plan['dataset_id'])
        for path in [self.output/'PLAN.json', self.output/'manifest.json', self.output/'catalog.json', self.output/'COMPLETE.json']:
            self.assertNotIn(str(self.raw), path.read_text())
        moved = self.root / 'relocated/raw'
        moved.parent.mkdir()
        shutil.move(self.raw, moved)
        for entry in self.manifest['files']:
            path = moved / entry['path']
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000))
        self.manifest['destination'] = str(moved)
        atomic_json(self.control / 'manifest.json', self.manifest)
        mark_all(self.control, self.manifest)
        relocated = self.root / 'relocated-index'
        second = index.prepare(self.control, relocated, rows_per_file=1)
        self.assertEqual(plan, second)
        index.worker(self.control, relocated, 'b')
        index.finalize(self.control, relocated)
        self.assertEqual((self.output/'catalog.json').read_bytes(), (relocated/'catalog.json').read_bytes())
        self.assertEqual((self.output/'COMPLETE.json').read_bytes(), (relocated/'COMPLETE.json').read_bytes())
        for row in rows:
            for ref in row['images']:
                self.assertEqual(read_ref_bytes(ref, roots={plan['dataset_id']: str(moved)}), self.payloads[ref['path']])

    def test_source_corruption_with_preserved_stat_cannot_commit(self):
        plan = self.prepare()
        relative = plan['units'][0]['relative_path']
        mark(self.control, self.manifest, relative)
        path = self.raw / relative
        stat = path.stat()
        with path.open('r+b') as stream:
            stream.seek(520)
            byte = stream.read(1)
            stream.seek(520)
            stream.write(bytes([byte[0] ^ 1]))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        current = index.worker(self.control, self.output, 'a')
        self.assertEqual(current['units_complete'], 0)
        self.assertEqual(current['unit_errors'], {'ValueError': 1})
        self.assertFalse((self.output/'units'/plan['units'][0]['unit_id']/'COMPLETE.json').exists())

    def test_resume_removes_only_uncommitted_outputs_and_checks_parquet(self):
        plan = self.prepare()
        directory = self.output / 'units' / plan['units'][0]['unit_id']
        directory.mkdir(parents=True)
        (directory/'crash.partial').write_bytes(b'uncommitted')
        before = {entry['path']: file_sha(self.raw/entry['path']) for entry in self.manifest['files']}
        mark_all(self.control, self.manifest)
        source_complete(self.control, self.manifest)
        index.worker(self.control, self.output, 'a')
        self.assertFalse((directory/'crash.partial').exists())
        self.assertEqual(before, {entry['path']: file_sha(self.raw/entry['path']) for entry in self.manifest['files']})
        receipt = json.loads((directory/'COMPLETE.json').read_text())
        parquet = self.output / receipt['catalog'][0]['path']
        payload = bytearray(parquet.read_bytes())
        payload[20] ^= 1
        parquet.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, 'Parquet changed'):
            index.finalize(self.control, self.output)
        self.assertFalse((self.output/'COMPLETE.json').exists())

    def test_bounded_buffers_and_safe_output_and_reserve(self):
        with patch.object(index, 'MAX_BUFFER_BYTES', 1):
            self.prepare()
            mark_all(self.control, self.manifest)
            result = index.worker(self.control, self.output, 'a', max_units=1)
        self.assertEqual(result['units_complete'], 1)
        receipt = next((self.output/'units').glob('*/COMPLETE.json'))
        self.assertEqual(len(json.loads(receipt.read_text())['catalog']), 2)
        with self.assertRaisesRegex(ValueError, 'at least 1 TiB'):
            index.worker(self.control, self.output, 'a', reserve_bytes=0)
        with self.assertRaisesRegex(ValueError, 'separate'):
            index.prepare(self.control, self.raw/'derived')

    def test_official_inventory_tamper_rejected_before_output(self):
        changed = deepcopy(self.manifest)
        changed['files'][0]['sha256'] = 'f' * 64
        atomic_json(self.control/'manifest.json', changed)
        with self.assertRaisesRegex(ValueError, 'official source'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_output_parent_symlinks_cannot_write_raw_dataset(self):
        before = {str(path.relative_to(self.raw)): file_sha(path)
                  for path in self.raw.rglob('*') if path.is_file()}
        self.output.mkdir(parents=True)
        (self.output/'units').symlink_to(self.raw, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.prepare()
        (self.output/'units').unlink()
        (self.output/'state').symlink_to(self.raw, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.prepare()
        self.assertEqual(before, {str(path.relative_to(self.raw)): file_sha(path)
                                 for path in self.raw.rglob('*') if path.is_file()})

    def test_lockf_excludes_another_process(self):
        context = multiprocessing.get_context('fork')
        queue = context.Queue()
        path = self.root / 'lockfiles/unit.lock'
        with index._lock(path) as acquired:
            self.assertTrue(acquired)
            process = context.Process(target=lock_probe, args=(str(path), queue))
            process.start()
            self.assertFalse(queue.get(timeout=10))
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        process = context.Process(target=lock_probe, args=(str(path), queue))
        process.start()
        self.assertTrue(queue.get(timeout=10))
        process.join(timeout=10)
        self.assertEqual(process.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
