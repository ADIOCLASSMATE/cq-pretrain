from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from .test_training_release import Fixture,sha
from . import training_release as training
from . import source_view as view
from . import selection_overlay as overlay
from corpus_pipeline.postprocess.reconcile import file_sha


class SelectionOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);source=self.root/'input';source.mkdir()
        fixture=Fixture(source);self.target=fixture.asset('placeholder')
        safe=fixture.asset('safe');near=fixture.asset('different_encoding','placeholder')
        bad=fixture.asset('bad',action='exclude')
        broken=fixture.asset('broken',action='pending',status='pending_integrity')
        fixture.part('mixed',[
            {'name':'target','images':[self.target],'text':'shirt'},
            {'name':'multi','images':[safe,self.target],'text':'book and shirt','category':'interleaved'},
            {'name':'repeat-target','images':[self.target,self.target],'text':'repeated placeholder'},
            {'name':'already-excluded','images':[self.target,bad],'text':'exclude'},
            {'name':'already-quarantined','images':[broken,self.target],'text':'quarantine'},
            {'name':'canonical-only','images':[near],'text':'different original SHA'},
            {'name':'pure','images':[],'text':'pure text','category':'pure_text'},
        ])
        fixture.part('untouched',[{'name':'safe','images':[safe],'text':'safe'}]);fixture.finish()
        self.candidate=self.root/'candidate';self.base_view=self.root/'base_view';self.output=self.root/'overlay'
        training.build(source,source/'assets.json',self.candidate,threads=1,memory_limit='256MB')
        view.build(self.candidate,self.base_view,threads=1,memory_limit='256MB')
        self.evidence=self.root/'evidence.json';self.evidence.write_text('{"confirmed":"placeholder"}\n')
        self.base_bytes={str(p):file_sha(p) for folder in [source,self.candidate,self.base_view]
                         for p in folder.rglob('*') if p.is_file()}
    def build(self,**kwargs):
        return overlay.build(self.candidate,self.base_view,self.output,asset_ids=[self.target],
            evidence=[self.evidence],threads=1,memory_limit='256MB',**kwargs)
    def test_exact_match_whole_record_exclusion_and_unchanged_native_labels(self):
        result=self.build();self.assertTrue(result['complete']);self.assertEqual(result['changed_records'],3)
        self.assertEqual(result['counts'],{'keep':3,'exclude':4,'quarantine':1})
        matches=pq.read_table(self.output/'matched_records.parquet').to_pylist()
        self.assertEqual(len(matches),5)
        self.assertNotIn(sha('canonical-only'),{x['sample_id'] for x in matches})
        changes=pq.read_table(self.output/'changed_records.parquet').to_pylist()
        self.assertEqual({x['sample_id'] for x in changes},{sha(x) for x in ('target','multi','repeat-target')})
        self.assertTrue(all(x['base_decision']=='keep' and x['overlay_decision']=='exclude' and x['overlay_reason']==overlay.REASON for x in changes))
        remaining=[row for batch in view.iter_native_batches(self.output/'source_view',batch_size=2) for row in batch.to_pylist()]
        self.assertEqual({x['sample_id'] for x in remaining},{sha(x) for x in ('canonical-only','pure','safe')})
        base=json.loads((self.base_view/'manifest.json').read_text());new=json.loads((self.output/'source_view/manifest.json').read_text())
        self.assertTrue(new['selection_overlay']);self.assertTrue(new['input_candidate_counts_are_base'])
        by_path={x['source_path']:x for x in base['files']}
        for item in new['files']:
            before=by_path[item['source_path']]
            for key in ('source_path','source_sha256','source_stamp','rows','mask_path'):
                self.assertEqual(item[key],before[key])
            if Path(item['source_path']).name=='untouched.parquet':
                self.assertEqual((self.base_view/item['mask_path']).stat().st_ino,(self.output/'source_view'/item['mask_path']).stat().st_ino)
        for path,expected in self.base_bytes.items():self.assertEqual(file_sha(path),expected)
        manifest=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(sum(x['records'] for x in manifest['source_category_counts']),8)
        self.assertEqual(sum(manifest['reason_counts'].values()),8)
        self.assertEqual(sum(x['changed_records'] for x in manifest['source_category_deltas']),3)
    def test_checkpoint_resume_and_output_tamper_rejected(self):
        partial=self.build(stop_after_buckets=2);self.assertFalse(partial['complete'])
        saved={str(p):file_sha(p) for p in (self.output/'matched_parts').glob('*/records.parquet')}
        result=self.build();self.assertEqual(self.build(),result)
        for path,expected in saved.items():self.assertEqual(file_sha(path),expected)
        manifest=json.loads((self.output/'source_view/manifest.json').read_text())
        item=next(x for x in manifest['files'] if Path(x['source_path']).name=='mixed.parquet')
        path=self.output/'source_view'/item['mask_path'];mask=np.fromfile(path,dtype=np.uint8);mask[0]=1;path.write_bytes(mask.tobytes())
        with self.assertRaisesRegex(ValueError,'mask byte count or SHA'):self.build()
    def test_evidence_and_base_tamper_fail_closed(self):
        self.build();before=self.evidence.read_bytes();self.evidence.write_bytes(before+b' ')
        with self.assertRaises(ValueError):self.build()
        # Restore the original file metadata is intentionally insufficient: its evidence binding is immutable.
        self.evidence.write_bytes(before)
        fresh=self.root/'other_overlay'
        overlay.build(self.candidate,self.base_view,fresh,asset_ids=[self.target],evidence=[self.evidence],threads=1,memory_limit='256MB')
        path=next(self.candidate.glob('buckets/*/keep.parquet'));path.write_bytes(path.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'Missing/changed output artifact'):
            overlay.build(self.candidate,self.base_view,fresh,asset_ids=[self.target],evidence=[self.evidence],threads=1,memory_limit='256MB')


if __name__=='__main__':unittest.main()
