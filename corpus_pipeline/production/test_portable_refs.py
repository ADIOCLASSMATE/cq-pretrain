"""Original TAR relocation, pinned-content integrity and legacy compatibility."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import portable_refs
from .records import load_raw_record, load_record, read_asset_bytes


DATASET = 'mlfoundations/dcvlm-baseline-200b@251523f5a04d7a852223454182cea63bc9f95063'


class ArchiveVerificationCacheTests(unittest.TestCase):
    def setUp(self):
        self.payload = b'archive bytes'
        self.state = SimpleNamespace(st_dev=1, st_ino=2, st_size=len(self.payload),
                                     st_mtime_ns=3, st_ctime_ns=4)
        self.value = {'dataset_id': DATASET, 'archive_size': len(self.payload),
                      'archive_sha256': hashlib.sha256(self.payload).hexdigest()}
        self.reads = 0
        self.enterContext(patch.object(portable_refs, '_ARCHIVE_CACHE', OrderedDict()))
        self.enterContext(patch.object(portable_refs.os, 'fstat', lambda fd: self.state))
        self.enterContext(patch.object(portable_refs.os, 'pread', self.pread))

    def pread(self, fd, size, offset):
        self.reads += 1
        return self.payload[offset:offset + size]

    def test_full_baseline_archive_population_reuses_first_verification(self):
        # Simulate all baseline TAR identities with tiny bytes, avoiding a real
        # 25 TB scan while exercising the real verifier and its eviction policy.
        paths = [f'/dataset/shards/{number:06d}.tar' for number in range(10752)]
        for path in paths:
            portable_refs._verify_archive(42, path, self.value)
        self.assertEqual(self.reads, len(paths))
        for path in (paths[0], paths[2048], paths[-1]):
            portable_refs._verify_archive(42, path, self.value)
        self.assertEqual(self.reads, len(paths), 'Previously verified TAR was unnecessarily rehashed')

    def test_cached_archive_rehashes_changed_stat_and_rejects_corruption(self):
        path = '/dataset/shards/000000.tar'
        portable_refs._verify_archive(42, path, self.value)
        self.assertEqual(self.reads, 1)
        for field in ('st_dev', 'st_ino', 'st_mtime_ns', 'st_ctime_ns'):
            with self.subTest(changed_stat=field):
                setattr(self.state, field, getattr(self.state, field) + 1)
                before = self.reads
                portable_refs._verify_archive(42, path, self.value)
                self.assertEqual(self.reads, before + 1)
                portable_refs._verify_archive(42, path, self.value)
                self.assertEqual(self.reads, before + 1)
        self.payload = b'X' + self.payload[1:]  # Same size, original expected SHA.
        self.state.st_ctime_ns += 1
        before = self.reads
        with self.assertRaisesRegex(ValueError, 'archive SHA256 mismatch'):
            portable_refs._verify_archive(42, path, self.value)
        self.assertEqual(self.reads, before + 1)
        # A failed verification must never install a successful cache entry.
        with self.assertRaisesRegex(ValueError, 'archive SHA256 mismatch'):
            portable_refs._verify_archive(42, path, self.value)
        self.assertEqual(self.reads, before + 2)


class PortableReferencesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / 'original-root'
        self.relative = 'shards/000/000000.tar'
        self.archive = self.old / self.relative
        self.archive.parent.mkdir(parents=True)
        self.payload = bytes(range(256)) * 17
        with tarfile.open(self.archive, 'w') as archive:
            for name, body in [('sample.0.jpg', self.payload), ('sample.conversations.txt', b'human:Hi<EOCL>\n')]:
                item = tarfile.TarInfo(name)
                item.size = len(body)
                archive.addfile(item, io.BytesIO(body))
        with tarfile.open(self.archive) as archive:
            item = archive.getmember('sample.0.jpg')
        self.ref = {'kind': 'tar', 'path': self.relative, 'member': item.name,
                    'offset': item.offset_data, 'offset_data': item.offset_data,
                    'length': item.size, 'size': item.size,
                    'sha256': hashlib.sha256(self.payload).hexdigest()}
        self.metadata = {**self.ref, 'dataset_id': DATASET,
                         'archive_size': self.archive.stat().st_size,
                         'archive_sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest()}
        self.ref['metadata_json'] = json.dumps(self.metadata)
        self.mapping = self.root / 'roots.json'
        self.mapping.write_text(json.dumps({DATASET: str(self.old)}))
        self.env = patch.dict(os.environ, CORPUS_DATASET_ROOTS=str(self.mapping))
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_large_metadata_range_keeps_separate_inference_limit(self):
        from corpus_pipeline.unified import ingest
        from corpus_pipeline.cluster import inference
        size = 147512032  # Actual pinned baseline member size, above 128 MiB.
        sparse = self.root/'large.tar'
        with sparse.open('wb') as handle:
            handle.truncate((256 << 20) + 1024)
        ref = dict(kind='tar', path=str(sparse), member='sample.3.jpg',
                   offset_data=512, size=size, sha256='a'*64)
        value = ingest._canonical_ref(ref)
        self.assertEqual(value['size'], size)
        self.assertEqual(portable_refs._range(value), (512, size))
        ref['size'] = 256 << 20
        self.assertEqual(ingest._canonical_ref(ref)['size'], 256 << 20)
        self.assertEqual(portable_refs._range(ref), (512, 256 << 20))
        ref['size'] += 1
        with self.assertRaises(ValueError):
            ingest._canonical_ref(ref)
        with self.assertRaises(ValueError):
            portable_refs._range(ref)
        self.assertEqual(inference.MAX_BYTES, 128 << 20)

    def test_move_original_root_only_and_change_mapping_preserves_bytes_and_reference(self):
        frozen = json.dumps(self.ref, sort_keys=True)
        before = self.archive.stat()
        self.assertEqual(read_asset_bytes(self.ref), self.payload)
        relocated = self.root / 'new-platform-mount'
        shutil.copytree(self.old, relocated)
        target = relocated / self.relative
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 10_000_000))
        self.assertNotEqual(target.stat().st_ino, before.st_ino)
        self.assertNotEqual(target.stat().st_mtime_ns, before.st_mtime_ns)
        shutil.rmtree(self.old)
        update = self.mapping.with_suffix('.new')
        update.write_text(json.dumps({DATASET: str(relocated)}))
        os.replace(update, self.mapping)
        self.assertEqual(read_asset_bytes(self.ref), self.payload)
        self.assertEqual(read_asset_bytes(self.ref, roots={DATASET: str(relocated)}), self.payload)
        self.assertEqual(json.dumps(self.ref, sort_keys=True), frozen)

    def test_traversal_absolute_path_and_symlink_escape_rejected(self):
        for path in ('../000000.tar', '/tmp/000000.tar', 'shards/../000000.tar', 'shards\\000000.tar'):
            bad = {**self.metadata, 'path': path}
            with self.subTest(path=path), self.assertRaises(ValueError):
                read_asset_bytes(bad)
        outside = self.root / 'outside.tar'
        shutil.copyfile(self.archive, outside)
        linked = self.old / 'linked.tar'
        linked.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            read_asset_bytes({**self.metadata, 'path': 'linked.tar'})

    def test_cached_archive_detects_same_size_tamper_even_when_mtime_restored(self):
        self.assertEqual(read_asset_bytes(self.ref), self.payload)
        before = self.archive.stat()
        # Some filesystems coarsen ctime updates to clock ticks. Ensure this
        # write actually changes the stat fingerprint used by the hash cache.
        time.sleep(.02)
        # Damage another member, outside the requested image: archive hash must
        # still reject this even though the requested image SHA remains correct.
        with self.archive.open('r+b') as handle:
            handle.seek(self.ref['offset'] + self.ref['length'] + 1024)
            byte = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([byte[0] ^ 1]))
        os.utime(self.archive, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.archive.stat().st_size, before.st_size)
        with self.assertRaisesRegex(ValueError, 'archive SHA256'):
            read_asset_bytes(self.ref)

    def test_requested_member_tamper_rejected_even_with_unchanged_cache_stamp(self):
        self.assertEqual(read_asset_bytes(self.ref), self.payload)
        frozen = portable_refs._stamp(self.archive.stat())
        with self.archive.open('r+b') as handle:
            handle.seek(self.ref['offset'])
            handle.write(b'X')
        # An unchanged filesystem fingerprint can never bypass the member hash.
        with patch.object(portable_refs, '_stamp', return_value=frozen):
            with self.assertRaisesRegex(ValueError, 'image SHA256'):
                read_asset_bytes(self.ref)

    def test_archive_hash_is_cached_but_member_hash_always_checked(self):
        actual = os.pread
        with patch.object(portable_refs.os, 'pread', wraps=actual) as reads:
            self.assertEqual(read_asset_bytes(self.ref), self.payload)
            initial_calls = reads.call_count
            self.assertGreater(initial_calls, 1)
            self.assertEqual(read_asset_bytes(self.ref), self.payload)
            self.assertEqual(reads.call_count - initial_calls, 1)
        bad = {**self.metadata, 'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'image SHA256'):
            read_asset_bytes(bad)

    def test_binding_size_range_and_allowlist_are_enforced(self):
        for change in ({'archive_size': self.metadata['archive_size'] + 1},
                       {'offset': 999999, 'offset_data': 999999},
                       {'offset': 1}, {'dataset_id': None}, {'member': '../escape.jpg'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_asset_bytes({**self.metadata, **change})
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            portable_refs.resolve_ref(self.ref, allowed_datasets=[])
        with self.assertRaisesRegex(ValueError, 'conflicting portable'):
            read_asset_bytes({**self.ref, 'path': 'another.tar'})
        for key, value in (('dataset_id', 'other/repository@' + '0' * 40),
                           ('archive_size', self.metadata['archive_size'] + 512),
                           ('archive_sha256', '0' * 64), ('sha256', '0' * 64),
                           ('member', 'other.0.jpg'), ('offset', self.ref['offset'] + 1)):
            with self.subTest(conflicting_field=key), self.assertRaisesRegex(ValueError, 'conflicting portable'):
                read_asset_bytes({**self.ref, key: value})

    def test_legacy_absolute_reference_and_npu_common_reader(self):
        legacy = {k: v for k, v in self.ref.items() if k != 'metadata_json'}
        legacy['path'] = str(self.archive)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_asset_bytes(legacy), self.payload)
        # The NPU resident worker resolves TAR references through this same
        # production reader before decoding; no accelerator/model is required.
        from corpus_pipeline.cluster.inference import _read_blob
        body, checksum = _read_blob({'sha256': self.ref['sha256'], 'original_ref': self.ref})
        self.assertEqual((body, checksum), (self.payload, self.ref['sha256']))

    def native_fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        metadata_root = self.root / 'metadata'
        metadata_root.mkdir()
        path = metadata_root / '000000.parquet'
        segments = [dict(kind='image', text=None, role='user', turn=0, image_index=0,
                         source_member=self.ref['member']),
                    dict(kind='text', text='Describe the image concisely.', role='user',
                         turn=0, image_index=None, source_member='sample.conversations.txt'),
                    dict(kind='text', text='A caption.', role='assistant', turn=1,
                         image_index=None, source_member='sample.conversations.txt')]
        raw = dict(collection='dcvlm', source='pixmo_cap', sample_id='sample',
                   source_shard=self.relative, source_key='sample', format='conversation',
                   segments=segments, images=[self.ref], image_count=1,
                   adapter_metadata_json='{}', original_locator_json=json.dumps({
                       'adapter': 'dcvlm.baseline_webdataset.v1', 'path': self.relative,
                       'dataset_id': DATASET, 'source_shard': self.relative, 'source_key': 'sample'}))
        pq.write_table(pa.Table.from_pylist([raw]), path)
        manifest = dict(schema_version='corpus.native.v2', allowed_image_roots=[], portable_datasets=[DATASET])
        snap = dict(root=str(metadata_root), manifest=manifest)
        spec = dict(parquet_path=str(path), parquet_row_index=0)
        return raw, snap, spec

    def test_native_v2_manifest_allows_portable_id_without_absolute_source_refs(self):
        raw, snap, spec = self.native_fixture()
        original = deepcopy(raw)
        with patch.dict(os.environ, {}, clear=True):
            result = load_raw_record(raw, spec, snap, {}, dataset_roots={DATASET: str(self.old)})
        self.assertEqual(result['source_image_refs'][0]['path'], self.relative)
        self.assertEqual(result['source_image_refs'][0]['dataset_id'], DATASET)
        self.assertEqual(result['normalized']['images'][0]['path'], self.relative)
        self.assertEqual(raw, original)
        snap['manifest']['portable_datasets'] = []
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            load_raw_record(raw, spec, snap, {})

    def test_mixed_source_tar_catalog_has_closed_source_membership(self):
        raw, snap, spec = self.native_fixture()
        path = Path(spec['parquet_path'])
        entry = dict(source='__mixed__', sources=['pixmo_cap', 'other_allowed_source'],
                     contract='mixed_source_tar.v1', source_shard=raw['source_shard'],
                     collection='dcvlm', bytes=path.stat().st_size)
        snap['catalog_index'] = {str(path): entry}
        snap['manifest']['catalog_contract'] = 'mixed_source_tar.v1'
        result = load_raw_record(raw, spec, snap, {})
        self.assertEqual(result['source'], 'pixmo_cap')
        for field, value in [('source', 'unlisted'), ('source_shard', 'shards/000/999999.tar'),
                             ('collection', 'genund')]:
            invalid = deepcopy(raw); invalid[field] = value
            with self.assertRaisesRegex(ValueError, 'catalog'):
                load_raw_record(invalid, spec, snap, {})
        for sources in ([], ['pixmo_cap', 'pixmo_cap'], ['__mixed__'], [None]):
            invalid = deepcopy(snap)
            invalid['catalog_index'][str(path)]['sources'] = sources
            with self.assertRaisesRegex(ValueError, 'membership'):
                load_raw_record(raw, spec, invalid, {})
        invalid = deepcopy(snap); invalid['catalog_index'][str(path)].pop('contract')
        with self.assertRaisesRegex(ValueError, 'membership'):
            load_raw_record(raw, spec, invalid, {})
        snap['manifest'].pop('catalog_contract')
        with self.assertRaisesRegex(ValueError, 'catalog'):
            load_raw_record(raw, spec, snap, {})

    def test_parquet_normalization_and_npu_reader_remain_identical_after_raw_root_moves(self):
        from corpus_pipeline.cluster.inference import _read_blob
        _, snap, spec = self.native_fixture()
        before = load_record(spec, snap, {})
        relocated = self.root / 'platform-copy'
        shutil.copytree(self.old, relocated)
        target = relocated / self.relative
        timestamp = target.stat()
        os.utime(target, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns + 100_000_000))
        shutil.rmtree(self.old)
        update = self.mapping.with_suffix('.next')
        update.write_text(json.dumps({DATASET: str(relocated)}))
        os.replace(update, self.mapping)
        after = load_record(spec, snap, {})
        self.assertEqual(after, before)
        for ref in (after['source_image_refs'][0], after['normalized']['images'][0]):
            self.assertEqual(ref['path'], self.relative)
            self.assertEqual(read_asset_bytes(ref), self.payload)
            blob, _ = _read_blob({'sha256': ref['sha256'], 'original_ref': ref})
            self.assertEqual(blob, self.payload)
        self.assertNotIn(str(self.old), json.dumps(after, sort_keys=True))
        self.assertNotIn(str(relocated), json.dumps(after, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
