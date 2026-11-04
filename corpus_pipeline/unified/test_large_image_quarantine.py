"""Real oversized/animated decode failures through the metadata release path.

Only the valid control image uses synthetic model evidence. Bad-image outcomes
come from the unmodified ResidentInferenceWorker._prepare, without loading NPU
models. No frame selection or image replacement is permitted by this fixture.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from corpus_pipeline.cluster import frontpass, inference, monitor, queue
from corpus_pipeline.dedup import asset_release, source_view, training_release
from corpus_pipeline.postprocess import reconcile, sscd_parallel_catalog
from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.production.records import snapshot
from . import baseline_index, native_compact
from .cache import CachedInferenceWorker
from .screening import build_decisions
from .test_baseline_index import fixture, mark_all, source_complete
from .test_cache import FakeWorker
from .test_ingest import image_bytes, write_tar


LARGE_BYTES = 147_512_032


def animation():
    stream = io.BytesIO()
    Image.new('RGB', (160, 128), 'red').save(stream, format='GIF', save_all=True,
        append_images=[Image.new('RGB', (160, 128), 'blue')], duration=100, loop=0)
    return stream.getvalue()


class DecodeDelegate(FakeWorker):
    def __init__(self, bad_ids):
        super().__init__()
        self.bad_ids = bad_ids
        self.decode_calls = []
        self.decoder = inference.ResidentInferenceWorker.__new__(inference.ResidentInferenceWorker)
        self.decoder.identity_sha256 = self.identity_sha256
        # Neither models nor torch exist: bad data must be rejected before them.

    def iter_results(self, assets, *, models=None):
        for asset in assets:
            if asset['sha256'] in self.bad_ids:
                self.decode_calls.append(asset['sha256'])
                prediction, tensors = self.decoder._prepare(asset, tuple(models or self.names))
                if tensors is not None or prediction.get('error_stage') != 'decode':
                    raise AssertionError('Bad animation unexpectedly reached model preprocessing')
                yield prediction
            else:
                yield from super().iter_results([asset], models=models)


class LargeImageQuarantineTests(unittest.TestCase):
    def test_large_bytes_reject_before_pillow_and_small_animation_rejects_before_transforms(self):
        worker = inference.ResidentInferenceWorker.__new__(inference.ResidentInferenceWorker)
        worker.identity_sha256 = 'a' * 64
        self.assertEqual(inference.MAX_BYTES, 128 << 20)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'oversized.png'
            with path.open('wb') as stream:
                stream.write(b'GIF89a'); stream.truncate(LARGE_BYTES)
            with patch.object(inference.Image, 'open', side_effect=AssertionError('Oversized input must not decode')) as opened:
                result, tensors = worker._prepare({'sha256': 'b' * 64, 'image_path': str(path)}, inference.MODEL_NAMES)
            opened.assert_not_called()
            self.assertIsNone(tensors)
            self.assertEqual(result['error_stage'], 'decode')
            self.assertEqual(result['error_type'], 'ValueError')
            self.assertIn('excessive image bytes', result['error'])
            self.assertEqual(result['scores'], {})
            self.assertNotIn('sscd', result)
        blob = animation()
        result, tensors = worker._prepare({'sha256': hashlib.sha256(blob).hexdigest(),
                                            'blob_loader': lambda: blob}, inference.MODEL_NAMES)
        self.assertIsNone(tensors)
        self.assertEqual(result['error_stage'], 'decode')
        self.assertIn('multiframe input requires a frame-specific adapter', result['error'])
        self.assertEqual(result['scores'], {})
        self.assertNotIn('sscd', result)

    def test_portable_large_animation_completes_queue_and_retains_quarantined_native_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, control, manifest, _ = fixture(root, count=1)
            small = animation(); large = small + b'\0' * (LARGE_BYTES - len(small))
            bad_ids = {hashlib.sha256(blob).hexdigest() for blob in (small, large)}
            examples = [('large-a', large), ('large-b', large), ('animated', small), ('valid', image_bytes())]
            members = []
            annotations = {}
            for name, blob in examples:
                text = 'human:  <image>Keep original animation and spaces.\n<EOCL>\ngpt:Original answer ' + name + '.\n'
                annotations[name] = text
                members.extend([(name + '.conversations.txt', text), (name + '.data_type.txt', 'instruction'),
                    (name + '.ds_name.txt', 'visual-source'), (name + '.orig_url.txt', 'https://example.test/original'),
                    (name + '.ntok.txt', '19'), (name + '.mdp.txt', '2'), (name + '.0.png', blob)])
            relative = 'shards/000/000000.tar'; tar = write_tar(raw / relative, members)
            del members, examples, large
            entry = manifest['files'][0]; entry.update(size=tar.stat().st_size, sha256=file_sha(tar))
            metadata = json.loads((control / 'dataset_metadata.json').read_text())
            metadata['siblings'][0].update(size=entry['size'], lfs={'sha256': entry['sha256']})
            atomic_json(control / 'dataset_metadata.json', metadata)
            manifest.update(total_bytes=sum(row['size'] for row in manifest['files']),
                            metadata_sha256=file_sha(control / 'dataset_metadata.json'))
            atomic_json(control / 'manifest.json', manifest); mark_all(control, manifest); source_complete(control, manifest)
            raw_sha = file_sha(tar); index = root / 'index'; native = root / 'native'
            with patch.object(baseline_index, 'EXPECTED_TRAINING_FILES', 1), patch.object(baseline_index, '_space'):
                plan = baseline_index.prepare(control, index)
                baseline_index.worker(control, index, 'fixture')
                baseline_index.finalize(control, index)
            with patch.object(native_compact.shutil, 'disk_usage',
                              return_value=shutil._ntuple_diskusage(4 << 40, 1 << 40, 3 << 40)):
                native_compact.run(index, native, workers=1)
            roots = root / 'source_roots.json'; atomic_json(roots, {plan['dataset_id']: str(raw)})
            catalog = json.loads((native / 'catalog.json').read_text())
            originals = [row for item in catalog for row in pq.ParquetFile(native / item['path']).read().to_pylist()]
            native_hashes = {item['path']: file_sha(native / item['path']) for item in catalog}
            self.assertEqual(len(originals), 4)
            for row in originals:
                self.assertEqual(json.loads(row['original_supervision_json'])['source_annotations']['conversations.txt'],
                                 annotations[row['source_key']])
                self.assertEqual(row['images'][0]['path'], relative)
                self.assertEqual(json.loads(row['images'][0]['metadata_json'])['dataset_id'], plan['dataset_id'])
            run = root / 'frontpass'; frontpass.prepare(run, dataset=native, rows_per_task=16384)
            delegate = DecodeDelegate(bad_ids); worker = CachedInferenceWorker(delegate, root / 'cache')
            with patch.dict(os.environ, {'CORPUS_DATASET_ROOTS': str(roots)}):
                claim = queue.claim(run, 'fixture')
                summary, artifacts = frontpass.process_task(run, claim, worker, snapshot(native),
                    frontpass.load_rules(), SimpleNamespace(entries={}), {'cf384': 1., 'unifd': 1.})
                queue.complete(run, claim['task_id'], claim['token'], artifacts=artifacts,
                    processed_records=summary['counts']['records'], processed_images=summary['counts']['image_occurrences'])
            progress = queue.status(run)
            self.assertTrue(progress['all_tasks_completed'])
            self.assertEqual(progress['counts'].get('failed', 0), 0)
            self.assertIsNone(queue.claim(run, 'no-retry'))
            self.assertEqual(len(delegate.decode_calls), 2)  # Repeated large SHA has two original occurrences.
            monitor.finalize(run, progress); worker.close()
            recon = root / 'recon'; result = reconcile.run(run, recon, threads=1, memory_limit='256MB')
            self.assertEqual(result['counts']['api_review_assets'], 0)
            self.assertEqual(result['counts']['sscd_usable_assets'], 1)
            assets = {row['asset_id']: row for path in recon.glob('buckets/*/assets.parquet')
                      for row in pq.ParquetFile(path).read().to_pylist()}
            for aid in bad_ids:
                self.assertEqual(assets[aid]['global_status'], 'pending_integrity')
                self.assertFalse(assets[aid]['sscd_usable'])
                self.assertFalse(assets[aid]['explicit_negative'])
                self.assertIsNone(assets[aid]['cf384_prob_max'])
                self.assertIsNone(assets[aid]['unifd_prob_max'])
            vectors = root / 'vectors'
            sscd_parallel_catalog.prepare(recon, vectors, block_rows=2)
            sscd_parallel_catalog.worker(vectors, workers=1)
            self.assertEqual(sscd_parallel_catalog.finalize(vectors)['vectors'], 1)
            screening = root / 'screening'; screen = build_decisions(recon, screening)
            self.assertTrue(screen['complete']); self.assertEqual(screen['review_candidates'], 0)
            asset_map = root / 'asset_map'
            asset_release.build(recon, screening / 'COMPLETE.json', asset_map, threads=1, memory='256MB')
            candidate = root / 'candidate'
            final = training_release.build(run, asset_map / 'COMPLETE.json', candidate, threads=1, memory_limit='256MB')
            self.assertEqual(final['counts'], {'keep': 1, 'quarantine': 3})
            quarantined = [row for path in candidate.glob('buckets/*/quarantine.parquet')
                           for row in pq.ParquetFile(path).read().to_pylist()]
            expected = {row['sample_id']: row for row in originals}
            self.assertEqual(len(quarantined), 3)
            for row in quarantined:
                self.assertEqual(row['decision_reason'], 'unresolved_image_integrity_or_missing_asset')
                self.assertEqual(row['source_record_sha256'], digest(expected[row['sample_id']]))
                self.assertTrue(row['integrity_quarantine'])
            view = root / 'view'; source_view.build(candidate, view, threads=1, memory_limit='256MB')
            kept = [row for batch in source_view.iter_native_batches(view) for row in batch.to_pylist()]
            self.assertEqual([row['source_key'] for row in kept], ['valid'])
            masks = json.loads((view / 'manifest.json').read_text())
            self.assertEqual(masks['counts']['quarantine'], 3)
            self.assertEqual(file_sha(tar), raw_sha)
            self.assertEqual(native_hashes, {item['path']: file_sha(native / item['path']) for item in catalog})


if __name__ == '__main__':
    unittest.main()
