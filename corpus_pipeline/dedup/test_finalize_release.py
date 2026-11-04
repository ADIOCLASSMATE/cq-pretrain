from pathlib import Path
import json
import tempfile
import unittest
from .asset_release import artifact
from .finalize_release import finalize,verify_audit_binding,verified_selection_overlay
from corpus_pipeline.postprocess.reconcile import file_sha


class FinalizeReleaseTests(unittest.TestCase):
    def test_stale_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ['assets','decisions']:
                (root/name).mkdir();(root/name/'COMPLETE.json').write_text('{}')
            inputs={str(root/n/'COMPLETE.json'):file_sha(root/n/'COMPLETE.json') for n in ['assets','decisions']}
            verify_audit_binding(root,'assets',{'inputs':inputs})
            (root/'assets/COMPLETE.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Audited input changed'):
                verify_audit_binding(root,'assets',{'inputs':inputs})

    def test_crash_after_complete_repairs_current_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'release';root.mkdir()
            for name in ['CODE.json','REPORT.md']:(root/name).write_text('evidence')
            manifest={'stages':{},'audits':{},'code_snapshot':artifact(root/'CODE.json'),
                      'report':artifact(root/'REPORT.md'),
                      'source_view':str(root/'quality_overlay/source_view'),
                      'selection_overlay_applied':True}
            (root/'manifest.json').write_text(json.dumps(manifest))
            complete={'complete':True,'manifest_sha256':file_sha(root/'manifest.json')}
            (root/'COMPLETE.json').write_text(json.dumps(complete))
            self.assertEqual(finalize(root),complete)
            pointer=json.loads((root.parent/'DEDUP_CURRENT.json').read_text())
            self.assertEqual(pointer['manifest']['sha256'],complete['manifest_sha256'])
            self.assertEqual(pointer['source_view'],manifest['source_view'])

    def test_overlay_binding_counts_and_stale_audit(self):
        from .test_training_release import Fixture
        from .training_release import build as build_candidates
        from .source_view import build as build_view
        from .selection_overlay import build as build_overlay
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'input';source.mkdir()
            fixture=Fixture(source);target=fixture.asset('placeholder');safe=fixture.asset('safe')
            fixture.part('data',[{'name':'bad','images':[target],'text':'wrong caption'},
                                 {'name':'good','images':[safe],'text':'correct caption'}]);fixture.finish()
            candidate=root/'training_candidate_index';view=root/'training_candidate_source_view'
            build_candidates(source,source/'assets.json',candidate,threads=1,memory_limit='256MB')
            build_view(candidate,view,threads=1,memory_limit='256MB')
            evidence=root/'evidence.json';evidence.write_text('{"confirmed":true}')
            output=root/'quality_overlay'
            build_overlay(candidate,view,output,asset_ids=[target],evidence=[evidence],threads=1,memory_limit='256MB')
            (root/'audits').mkdir()
            required=[output/'COMPLETE.json',output/'manifest.json',
                      output/'source_view/COMPLETE.json',output/'source_view/manifest.json',evidence]
            audit={'passed':True,'inputs':{str(p):file_sha(p) for p in required}}
            audit_path=root/'audits/quality_overlay.json';audit_path.write_text(json.dumps(audit))
            base=json.loads((candidate/'manifest.json').read_text())
            (root/'PUBLISH_HOLD.json').write_text(json.dumps({'required_stage':'quality_overlay',
                'required_audit':'audits/quality_overlay.json','asset_ids':[target]}))
            result,_=verified_selection_overlay(root,candidate,view,base)
            self.assertEqual(result['final_counts'],{'keep':1,'exclude':1,'quarantine':0})
            audit['inputs'][str(output/'source_view/manifest.json')]='0'*64
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'audit belongs to another stage'):
                verified_selection_overlay(root,candidate,view,base)

    def test_unmet_publication_hold_cannot_repair_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'release';root.mkdir()
            for name in ['CODE.json','REPORT.md']:(root/name).write_text('evidence')
            manifest={'stages':{},'audits':{},'code_snapshot':artifact(root/'CODE.json'),
                      'report':artifact(root/'REPORT.md'),'source_view':str(root/'base')}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'COMPLETE.json').write_text(json.dumps({'complete':True,'manifest_sha256':file_sha(root/'manifest.json')}))
            (root/'PUBLISH_HOLD.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Publication hold requires quality overlay'):finalize(root)
            self.assertFalse((root.parent/'DEDUP_CURRENT.json').exists())


if __name__=='__main__':unittest.main()
