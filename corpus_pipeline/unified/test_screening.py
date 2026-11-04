import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from corpus_pipeline.dedup import decisions as old
from .screening import build_decisions


def qrow(n, action='admit', assessment='likely_non_ai', state='done'):
    primary = {'decision': action, 'assessment': assessment, 'confidence': 'high',
               'reason': 'ai_visual_review_high_confidence_generated' if action == 'reject' else 'visual_review'}
    return {'asset_id': f'{n:064x}', 'input_sha256': str(n), 'state': state,
            'decision_json': json.dumps({'input_sha256': str(n), 'model_recommendation': action,
                                        'reviews': {'primary': primary}})}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def recon(root, ids):
    root.mkdir(); rows = [{'asset_id': f'{n:064x}'} for n in ids]
    schema = pa.schema([('asset_id', pa.string())])
    for name, data in [('api_review', rows), ('assets', [{'asset_id': f'{n:064x}'} for n in range(1, 5)])]:
        pq.write_table(pa.Table.from_pylist(data, schema=schema), root/(name+'.parquet'))
    artifacts = [{**old.bound(root/(name+'.parquet')), 'path': name+'.parquet'} for name in ['api_review','assets']]
    write_json(root/'bucket.json', {'bucket': '00', 'artifacts': artifacts})
    write_json(root/'COMPLETE.json', {'complete': True, 'counts': {'api_review_assets': len(ids)},
        'buckets': [{'receipt_path': 'bucket.json', 'receipt_sha256': old.file_sha(root/'bucket.json')}]})


def review(root, reconciliation, rows):
    root.mkdir(); pq.write_table(pa.Table.from_pylist(rows), root/'decisions.parquet')
    write_json(root/'COMPLETE.json', {'screening_pass_complete': True,
        'scope': {'reconciliation_complete_sha256': old.file_sha(reconciliation/'COMPLETE.json')},
        'artifacts': [{'path': 'decisions.parquet', 'sha256': old.file_sha(root/'decisions.parquet')}]})


def history(root, rows):
    root.mkdir(); pq.write_table(pa.Table.from_pylist(rows, schema=old.ARROW_SCHEMA), root/'screening.parquet')
    write_json(root/'COMPLETE.json', {'schema': old.SCHEMA, 'complete': True,
        'files': [{**old.bound(root/'screening.parquet'), 'path': 'screening.parquet'}]})
    return root/'COMPLETE.json'


class ScreeningTests(unittest.TestCase):
    def test_empty_queue_is_complete_and_record_quality_stays_record_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);recon(p/'recon', [])
            rule={'scope':'source_record','source_record_sha256':'f'*64,'reason':'bad_caption',
                  'confirmed':True,'evidence_ids':['verified-original-record']}
            result=build_decisions(p/'recon',p/'out',quality_decisions=[rule])
            self.assertTrue(result['complete']);self.assertEqual(result['records'],0)
            self.assertEqual(len(result['record_quality_files']),1)
            self.assertEqual(pq.read_table(p/'out/record_quality.parquet')['source_record_sha256'].to_pylist(),['f'*64])

    def test_missing_results_are_not_completed_and_uncertain_final_review_is_annotated(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);recon(p/'recon',[1])
            pending=build_decisions(p/'recon',p/'out')
            self.assertFalse(pending['complete']);self.assertEqual(pending['pending_infra'],1)
            self.assertFalse((p/'out/COMPLETE.json').exists())
            review(p/'review',p/'recon',[qrow(1,'pending','uncertain')])
            done=build_decisions(p/'recon',p/'out',p/'review')
            self.assertTrue(done['complete']);self.assertEqual(done['counts'],{'pending':1})
            self.assertEqual(build_decisions(p/'recon',p/'out',p/'review'),done)
            with (p/'out/screening.parquet').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):build_decisions(p/'recon',p/'out',p/'review')

    def test_history_is_exact_scoped_and_quality_does_not_change_ai_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);recon(p/'recon',[1,2])
            h=history(p/'history',[old.normalize_qwen(qrow(n)) for n in [1,2,3,4]])
            rule={'scope':'asset','asset_id':f'{3:064x}','reason':'placeholder','confirmed':True,'evidence_ids':['image']}
            done=build_decisions(p/'recon',p/'out',quality_decisions=[rule],history_decision_manifests=[h])
            self.assertEqual(done['records'],3)
            rows={r['asset_id']:r for r in pq.read_table(p/'out/screening.parquet').to_pylist()}
            self.assertNotIn(f'{4:064x}',rows);self.assertEqual(rows[f'{3:064x}']['screen_action'],'exclude')
            self.assertEqual(rows[f'{3:064x}']['ai_decision'],'not_inferred_from_quality')
            self.assertEqual(done['historical_exclusions_without_current_alert'],0)

    def test_conflicting_history_and_failed_api_cannot_fake_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);recon(p/'recon',[1])
            first=history(p/'h1',[old.normalize_qwen(qrow(1))])
            second=history(p/'h2',[old.normalize_qwen(qrow(1,'reject','likely_ai_generated'))])
            result=build_decisions(p/'recon',p/'out',history_decision_manifests=[first,second])
            self.assertFalse(result['complete']);self.assertEqual(result['conflicts'],1)
            review(p/'review',p/'recon',[qrow(1,state='failed')])
            result=build_decisions(p/'recon',p/'failed',p/'review')
            self.assertFalse(result['complete']);self.assertEqual(result['pending_infra'],1)

    def test_confirmed_history_applies_without_new_alert_and_out_of_scope_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);recon(p/'recon',[])
            history_path=history(p/'history',[old.normalize_qwen(qrow(n,'reject','likely_ai_generated')) for n in [1,9]])
            absent_quality={'scope':'asset','asset_id':f'{8:064x}','reason':'placeholder','confirmed':True,'evidence_ids':['quality']}
            result=build_decisions(p/'recon',p/'out',history_decision_manifests=[history_path],quality_decisions=[absent_quality])
            self.assertTrue(result['complete']);self.assertEqual(result['records'],1)
            self.assertEqual(result['counts'],{'exclude':1});self.assertEqual(result['out_of_scope_assets'],2)
            self.assertEqual(result['historical_exclusions_without_current_alert'],1)


if __name__=='__main__':unittest.main()
