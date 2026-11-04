import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import atomic_json, artifact, digest, file_sha, stamp
from .training_release import text_fingerprint
from .asset_supervision_view import build, iter_assets, iter_supervisions, iter_asset_supervisions


def fixture(root):
    candidate = root / 'candidate'
    candidate.mkdir()
    first, equivalent, nearby = [digest(x) for x in ('first', 'same exact pixels', 'different crop')]
    specs = [('first', [first], [first], 'Caption one', 'single_i2t'),
             ('second', [equivalent], [first], 'Caption two', 'single_i2t'),
             ('duplicate', [equivalent], [first], 'Caption one', 'single_i2t'),
             ('nearby', [nearby], [nearby], 'Caption one', 'single_i2t'),
             ('generation', [first], [first], 'Caption one', 'single_t2i'),
             ('interleaved', [first, nearby, first], [first, nearby, first], 'Ordered dialogue', 'interleaved')]
    native = []
    for name, images, canonical, text, category in specs:
        segments = [{'kind': 'image', 'role': 'user', 'turn': 0, 'text': None, 'image_index': index}
                    for index in range(len(images))]
        segments.append({'kind': 'text', 'role': 'assistant', 'turn': 0, 'text': text, 'image_index': None})
        if category == 'single_t2i':
            segments = segments[-1:] + segments[:-1]
        native.append({'sample_id': digest(name), 'source': 'fixture', 'format': 'caption',
                       'image_count': len(images), 'text_hash': text_fingerprint(segments), 'segments': segments,
                       'images': [{'sha256': aid, 'kind': 'file', 'path': '/unread-fixture/' + aid + '.png'} for aid in images],
                       'annotation_field': name})
    source = root / 'original.parquet'
    pq.write_table(pa.Table.from_pylist(native), source)
    native = pq.read_table(source).to_pylist()
    decisions = []
    for index, ((name, images, canonical, text, category), raw) in enumerate(zip(specs, native)):
        decisions.append({'record_id': name, 'record_signature': digest([category, raw['text_hash'], canonical]),
                          'category': category, 'image_ids': images, 'canonical_image_ids': canonical,
                          'original_parquet_path': str(source), 'row_index': index,
                          'source_record_sha256': digest(raw), 'source': 'fixture',
                          'decision': 'exclude' if name == 'duplicate' else 'keep',
                          'decision_reason': 'identical_ordered_supervision_and_confirmed_images' if name == 'duplicate' else 'retain_candidate',
                          'duplicate_of_record_id': 'first' if name == 'duplicate' else None,
                          'unresolved_screening': False})
    bucket = candidate / 'records.parquet'
    pq.write_table(pa.Table.from_pylist(decisions), bucket)
    binding = candidate / 'binding.json'
    atomic_json(binding, {'path': str(source), 'stamp': stamp(source), 'sha256': file_sha(source)})
    manifest = {'scope': 'full_run', 'same_image_different_supervision_preserved': True,
                'sscd_candidates_used_for_deletion': False, 'records': len(native),
                'counts': {'keep': 5, 'exclude': 1},
                'buckets': [{'artifacts': [artifact(bucket, candidate)]}],
                'native_metadata_bindings': [artifact(binding, candidate)]}
    atomic_json(candidate / 'manifest.json', manifest)
    atomic_json(candidate / 'COMPLETE.json', {'complete': True, 'manifest_sha256': file_sha(candidate / 'manifest.json')})
    return candidate, first, nearby, native


class AssetSupervisionViewTests(unittest.TestCase):
    def test_explicit_i2t_route_is_preserved_for_interleaved_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);candidate,first,_,native=fixture(root)
            path=candidate/'records.parquet';rows=pq.read_table(path).to_pylist()
            for row in rows:
                row.update(processing_policy=None,task_route=None,image_loss_allowed=None)
                if row['category']=='interleaved':
                    row.update(processing_policy='corpus.task_policy.v1',task_route='i2t',image_loss_allowed=False)
            pq.write_table(pa.Table.from_pylist(rows),path)
            manifest=json.loads((candidate/'manifest.json').read_text())
            manifest['buckets'][0]['artifacts']=[artifact(path,candidate)]
            atomic_json(candidate/'manifest.json',manifest)
            atomic_json(candidate/'COMPLETE.json',{'complete':True,'manifest_sha256':file_sha(candidate/'manifest.json')})
            output=root/'view';build(candidate,output,threads=1,memory_limit='256MB')
            task=next(row for row in iter_asset_supervisions(output,first) if row['category']=='interleaved')
            self.assertEqual(task['direction'],'i2t')
            self.assertFalse(task['image_loss_allowed'])
            self.assertEqual(task['segments'],native[-1]['segments'])

    def test_single_asset_multiple_actual_supervisions_and_duplicate_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, first, nearby, native = fixture(root)
            source_hash = file_sha(root / 'original.parquet')
            output = root / 'view'
            result = build(candidate, output, threads=1, memory_limit='256MB')
            self.assertEqual(result['counts'], {'assets': 2, 'supervisions': 5,
                                               'asset_supervisions': 6, 'supervision_provenance': 6})
            assets = list(iter_assets(output))
            self.assertEqual({row['canonical_asset_id'] for row in assets}, {first, nearby})
            shared = next(row for row in assets if row['canonical_asset_id'] == first)
            self.assertEqual(shared['representative_original_asset_id'], first)
            self.assertEqual(shared['supervision_count'], 4)
            tasks = list(iter_asset_supervisions(output, first))
            self.assertEqual(len(tasks), 4)
            self.assertEqual({task['segments'][-1]['text'] for task in tasks if task['direction'] == 'i2t'},
                             {'Caption one', 'Caption two'})
            multi = next(task for task in tasks if task['category'] == 'interleaved')
            self.assertEqual(multi['ordered_image_ids'], [first, nearby, first])
            self.assertEqual(multi['image_indices'], [0, 2])
            self.assertEqual(multi['segments'], native[-1]['segments'])
            all_tasks = list(iter_supervisions(output))
            self.assertEqual(len(all_tasks), 5)
            for task in all_tasks:
                self.assertEqual(task['image_loss_allowed'], task['direction'] == 't2i')
                self.assertEqual(task['native_record'], native[task['row_index']])
            provenance = pq.read_table(output / 'supervision_provenance.parquet').to_pylist()
            first_task = next(task for task in tasks if task['record_id'] == 'first')
            self.assertEqual({row['record_id'] for row in provenance
                              if row['supervision_id'] == first_task['supervision_id']}, {'first', 'duplicate'})
            self.assertEqual(file_sha(root / 'original.parquet'), source_hash)
            self.assertEqual(build(candidate, output, threads=1, memory_limit='256MB'), result)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertFalse(manifest['image_payloads_copied'])
            self.assertFalse(manifest['training_sampler_changed'])

    def test_tampered_output_and_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _, _, _ = fixture(root)
            output = root / 'view'
            build(candidate, output, threads=1, memory_limit='256MB')
            (output / 'assets.parquet').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                list(iter_assets(output))
            (root / 'original.parquet').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                build(candidate, root / 'another', threads=1, memory_limit='256MB')


if __name__ == '__main__':
    unittest.main()
