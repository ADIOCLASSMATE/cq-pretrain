import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from corpus_pipeline.detector_calibration.sample_dcvlm import contained, domain_for, extract_bytes, random_rows, split_for, weighted_shards


class SamplingTest(unittest.TestCase):
    def test_byte_range_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'part.tar').write_bytes(b'headerIMAGEtail')
            ref = {'offset': 6, 'length': 5, 'sha256': hashlib.sha256(b'IMAGE').hexdigest()}
            self.assertEqual(extract_bytes(root, 'part.tar', ref), b'IMAGE')
            with self.assertRaises(ValueError):
                extract_bytes(root, 'part.tar', {**ref, 'sha256': 'bad'})
            with self.assertRaises(ValueError):
                extract_bytes(root, 'part.tar', {**ref, 'offset': -1})
            with self.assertRaises(ValueError):
                extract_bytes(root, '../escape.tar', ref)

    def test_repeatable_shard_and_split_selection(self):
        items = [{'path': str(i), 'rows': i + 1} for i in range(100)]
        a = weighted_shards(items, 3, random.Random(20261022))
        b = weighted_shards(list(reversed(items)), 3, random.Random(20261022))
        self.assertEqual(a, b)
        self.assertEqual(len(set(x['path'] for x in a)), 3)
        self.assertNotEqual([x['path'] for x in a], ['0', '1', '2'])
        self.assertEqual(split_for('abc'), split_for('abc'))

    def test_non_head_rows_across_row_groups(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.parquet"
            pq.write_table(pa.table({"ordinal": list(range(100))}), path, row_group_size=20)
            selected = list(random_rows(path, [7, 31, 49, 95]))
            self.assertEqual([row[0] for row in selected], [7, 31, 49, 95])
            self.assertEqual([row[1]["ordinal"] for row in selected], [7, 31, 49, 95])

    def test_domain_is_not_authenticity(self):
        self.assertEqual(domain_for('datacomp_1b'), 'mixed_web')
        self.assertEqual(domain_for('relaion-2b-en-vol-1-unpacked-TEST'), 'mixed_web')
        self.assertEqual(domain_for('clevr_math'), 'rendering_screenshot')
        self.assertEqual(domain_for('not_known'), 'other_visual')


if __name__ == '__main__':
    unittest.main()
