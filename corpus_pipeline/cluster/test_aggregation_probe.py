import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .aggregation_probe import MAJOR_SOURCES, profiles, reference_cases, selected_cases, warm_inputs_for
from .benchmark_launcher import MODULES


class AggregationProbeTests(unittest.TestCase):
    config = dict(batch_size=64, decode_workers=16, cpu_threads=1,
                  prefetch_batches=2, fast_prepare=True)

    def test_major_sources_cover_universal_and_large_batch_candidates(self):
        for source in MAJOR_SOURCES:
            values = profiles(source, self.config)
            self.assertEqual([(v['configuration']['batch_size'], v['configuration']['decode_workers'],
                               v['configuration']['cpu_threads']) for v in values[:3]],
                             [(32, 8, 2), (64, 16, 1), (32, 4, 4)])
            self.assertEqual([v['configuration']['batch_size'] for v in values[3:]],
                             [128, 256, 512] if source == 'datacomp_1b' else [128, 256])
            if source == 'datacomp_1b':
                self.assertEqual(values[-1]['configuration']['prefetch_batches'], 1)
            self.assertTrue(all(v['aggregation'] == dict(target_images=1024, max_records=1024,
                                                        max_record_bytes=64 << 20) for v in values))
            self.assertTrue(all(v['configuration']['fast_prepare'] for v in values))

    def test_edge_grids_have_unique_tokens_and_preserve_input(self):
        before = copy.deepcopy(self.config)
        for source, count in [('as100m', 2), ('anyword3m', 3), ('allava', 1)]:
            values = profiles(source, self.config)
            self.assertEqual(len(values), count)
            self.assertEqual(len({v['name'] for v in values}), count)
            self.assertTrue(all(v['aggregation']['target_images'] == 1024 for v in values))
        self.assertEqual(self.config, before)

    def test_unbounded_config_or_unknown_source_rejected(self):
        for config in ({**self.config, 'batch_size': 1024}, {**self.config, 'cpu_threads': True}):
            with self.assertRaises(ValueError):
                profiles('as100m', config)
        with self.assertRaises(ValueError):
            profiles('unknown', self.config)

    def test_reference_requires_exact_case_and_rows(self):
        selected = [('native-task', dict(source='as100m', row_start=0, row_end=2048))]
        baseline = dict(counts=dict(records=2048), elapsed_seconds=20.)
        reference = dict(success=True, dtype='float32', results=[dict(case='native-task', repetition=0,
            source='as100m', timings=dict(baseline=baseline), comparison=dict(record_outcomes_equal=True))])
        self.assertEqual(reference_cases(reference, selected), {'native-task': baseline})
        for mutate in (lambda r: r['results'][0].update(case='other-task'),
                       lambda r: r['results'][0]['timings']['baseline']['counts'].update(records=1024),
                       lambda r: r['results'].append(copy.deepcopy(r['results'][0])),
                       lambda r: r.update(success=False)):
            invalid = copy.deepcopy(reference)
            mutate(invalid)
            with self.assertRaises(ValueError):
                reference_cases(invalid, selected)

    def test_major_cases_keep_actual_source_file_fingerprint(self):
        import json
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'states').mkdir()
            catalog = {}
            for index, source in enumerate(MAJOR_SOURCES):
                path = root / (str(index) + '.parquet')
                path.write_bytes(b'file identity')
                catalog[str(path)] = dict(source=source, source_shard='shard-' + str(index), rows=2048)
            for task_id, source in [('s00030-r00000000', 'as100m'),
                                    ('s00016-r00000000', 'anyword3m'), ('s00007-r00000000', 'allava')]:
                (root / 'states' / (task_id + '.json')).write_text(json.dumps(dict(task=dict(
                    task_id=task_id, source=source, row_start=100, row_end=20000))))
            values = selected_cases(root, dict(catalog_index=catalog))
            self.assertEqual(len(values), 6)
            for _, task in values[:3]:
                self.assertEqual(task['row_end'], 1024)
                stat = Path(task['parquet_path']).stat()
                self.assertEqual(task['parquet_fingerprint'], dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            self.assertEqual([task['row_end'] - task['row_start'] for _, task in values[3:]], [2048, 1024, 8192])

    def test_bounded_launcher_accepts_probe(self):
        self.assertIn('aggregation_probe', MODULES)

    def test_sparse_warmup_uses_full_bounded_task_and_stops_lazily(self):
        task = dict(source='allava', row_start=100, row_end=8292)
        snap = dict(root='snapshot', fingerprint='identity')
        visited = []
        def records(manifest, snapshot, rules):
            self.assertEqual(manifest['shards'][0]['row_stop'], task['row_end'])
            self.assertEqual(manifest['shards'][0]['row_start'], task['row_start'])
            for index in range(1000):
                visited.append(index)
                yield {'source_image_refs': [{'sha256': 'image-' + str(index)}] if index >= 100 else []}
        with patch('corpus_pipeline.production.ingest.iter_records', records):
            result = warm_inputs_for(task, 4, snap, {})
        self.assertEqual([x['sha256'] for x in result], ['image-100', 'image-101', 'image-102', 'image-103'])
        self.assertEqual(visited, list(range(104)))

    def test_warmup_multi_image_record_and_insufficient_images(self):
        task = dict(source='test', row_start=0, row_end=2)
        snap = dict(root='snapshot', fingerprint='identity')
        records = [{'source_image_refs': [{'sha256': str(n)} for n in range(3)]}, {'source_image_refs': []}]
        with patch('corpus_pipeline.production.ingest.iter_records', return_value=iter(records)):
            self.assertEqual(len(warm_inputs_for(task, 2, snap, {})), 2)
        with patch('corpus_pipeline.production.ingest.iter_records', return_value=iter(records)):
            with self.assertRaisesRegex(ValueError, 'Insufficient images'):
                warm_inputs_for(task, 4, snap, {})


if __name__ == '__main__':
    unittest.main()
