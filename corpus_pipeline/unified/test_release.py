import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from corpus_pipeline.postprocess.reconcile import atomic_json, file_sha, stamp, digest
from .release import REQUIRED_STAGES, artifact, finalize, iter_batches, read_manifest, seal_receipt, _verify_full_native_snapshot


class ReleaseTests(unittest.TestCase):
    def compacted_fixture(self, root):
        from . import native_compact
        from .test_native_compact import fixture
        index = fixture(root)
        output = root / 'native'
        with patch.object(native_compact.shutil, 'disk_usage',
                          return_value=shutil._ntuple_diskusage(4 << 40, 1 << 40, 3 << 40)):
            native_compact.run(index, output, workers=2)
        return output

    def compacted_stage(self, output):
        from .workflow import finalized_files
        catalog = json.loads((output / 'catalog.json').read_text())
        collections = {'dcvlm': digest({'snapshot': file_sha(output / 'COMPLETE.json'),
                                       'collection': 'dcvlm', 'files': catalog})}
        stage = {'artifacts': [artifact(path) for path in finalized_files(output)],
                 'coverage': {'processed': 6}}
        return stage, collections

    def test_compacted_full_snapshot_accepts_real_workflow_lineage_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.compacted_fixture(Path(temp))
            stage, collections = self.compacted_stage(output)
            native_completions = [item['path'] for item in stage['artifacts']
                if Path(item['path']).name == 'COMPLETE.json'
                and json.loads(Path(item['path']).read_text()).get('schema_version') == 'corpus.native.v2']
            self.assertEqual(set(native_completions), {str(output / 'COMPLETE.json'),
                                                     str(output / 'lineage/COMPLETE.json')})
            _verify_full_native_snapshot(stage, collections)

    def test_parallel_primary_snapshots_still_rejected_with_valid_lineages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = self.compacted_fixture(root)
            second = root / 'another-native'; shutil.copytree(output, second)
            stage, collections = self.compacted_stage(output)
            other, _ = self.compacted_stage(second)
            stage['artifacts'] += other['artifacts']
            with self.assertRaisesRegex(ValueError, 'one bound native v2 snapshot'):
                _verify_full_native_snapshot(stage, collections)

    def test_lineage_cannot_promote_unvalidated_input_even_after_rebinding(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.compacted_fixture(Path(temp))
            source_path = output / 'lineage/COMPLETE.json'
            source_complete = json.loads(source_path.read_text()); source_complete['validation_passed'] = False
            atomic_json(source_path, source_complete)
            manifest_path = output / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
            manifest['source_lineage']['artifacts']['COMPLETE.json'].update(
                bytes=source_path.stat().st_size, sha256=file_sha(source_path))
            atomic_json(manifest_path, manifest)
            complete_path = output / 'COMPLETE.json'; complete = json.loads(complete_path.read_text())
            complete['manifest_sha256'] = file_sha(manifest_path); atomic_json(complete_path, complete)
            stage, collections = self.compacted_stage(output)
            with self.assertRaisesRegex(ValueError, 'fully verified pinned index'):
                _verify_full_native_snapshot(stage, collections)

    def test_lineage_metadata_is_required_even_when_nested_complete_not_listed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.compacted_fixture(Path(temp))
            stage, collections = self.compacted_stage(output)
            stage['artifacts'] = [item for item in stage['artifacts'] if '/lineage/' not in item['path']]
            # A manifest binds its transitive provenance even if an older stage
            # receipt did not enumerate those copied metadata files separately.
            _verify_full_native_snapshot(stage, collections)
            (output / 'lineage/PLAN.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'lineage bytes changed'):
                _verify_full_native_snapshot(stage, collections)

    def fixture(self, root, single_collection=False):
        source=root/'native.parquet'
        rows=[{'collection_id':'dcvlm','category':'multi_i2t','text':'question and answer','image_ids':['a'*64]},
              {'collection_id':'genund','category':'single_t2i','text':'a distinct caption','image_ids':['a'*64]},
              {'collection_id':'dcvlm','category':'single_i2t','text':'confirmed bad caption','image_ids':['a'*64]}]
        if single_collection:rows[1]['collection_id']='dcvlm'
        pq.write_table(pa.Table.from_pylist(rows),source)
        view=root/'view';view.mkdir();mask=view/'mask.u8';np.array([1,1,0],np.uint8).tofile(mask)
        counts={'keep':2,'exclude':1,'quarantine':0};masked={**counts,'not_indexed':0}
        manifest={'complete':True,'records':3,'native_rows':3,'mask_bytes':3,'source_files':1,
            'codes':{'exclude':0,'keep':1,'quarantine':2,'not_indexed':255},'counts':masked,
            'original_records_modified':False,'files':[{'source_path':str(source),'source_sha256':file_sha(source),
                'source_stamp':stamp(source),'rows':3,'mask_bytes':3,'mask_path':'mask.u8','mask_sha256':file_sha(mask),'counts':masked}]}
        atomic_json(view/'manifest.json',manifest)
        atomic_json(view/'COMPLETE.json',{'complete':True,'records':3,'counts':masked,'manifest_sha256':file_sha(view/'manifest.json')})
        collections={'dcvlm':'1'*64,'genund':'2'*64};coverage={'unit':'records','expected':3,'processed':3,'missing':0,'pending_infra':0}
        if single_collection:collections.pop('genund')
        receipt_paths={name:root/(name+'.json') for name in ['normalize','records','source_view']}
        seal_receipt(receipt_paths['normalize'],stage='normalize',collections=collections,coverage=coverage,artifacts=[source])
        statistics={'records':3,'counts':counts,'per_collection':{
            'dcvlm':{'records':2,'counts':{'keep':1,'exclude':1,'quarantine':0}},
            'genund':{'records':1,'counts':{'keep':1,'exclude':0,'quarantine':0}}}}
        if single_collection:statistics['per_collection']={'dcvlm':{'records':3,'counts':counts}}
        seal_receipt(receipt_paths['records'],stage='records',collections=collections,coverage=coverage,
            artifacts=[source],dependencies={'normalize':file_sha(receipt_paths['normalize'])},statistics=statistics,
            semantics={'task_direction_preserved':True,'ordered_supervision_preserved':True})
        seal_receipt(receipt_paths['source_view'],stage='source_view',collections=collections,coverage=coverage,
            artifacts=[view/'COMPLETE.json',view/'manifest.json'],dependencies={'records':file_sha(receipt_paths['records'])},
            reader={'kind':'native_source_view','path':str(view)})
        return collections,receipt_paths,rows

    def test_joint_release_preserves_directions_and_captions_and_repairs_current(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,rows=self.fixture(p)
            args=(p/'run',receipts)
            result=finalize(*args,expected_collections=collections,required_stages=receipts)
            actual=[row for batch in iter_batches(p/'run') for row in batch.to_pylist()]
            self.assertEqual(actual,rows[:2])
            self.assertEqual(result['counts'],{'keep':2,'exclude':1,'quarantine':0})
            self.assertEqual(result['scope'],'validation')
            (p/'run/CURRENT.json').unlink()
            self.assertEqual(finalize(*args,expected_collections=collections,required_stages=receipts),result)
            self.assertTrue((p/'run/CURRENT.json').exists())

    def test_corrupt_mask_and_missing_collection_block_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,_=self.fixture(p)
            with self.assertRaisesRegex(ValueError,'collection snapshot'):
                finalize(p/'run',receipts,expected_collections={'dcvlm':'1'*64},required_stages=receipts)
            np.array([0,1,1],np.uint8).tofile(p/'view/mask.u8')
            with self.assertRaisesRegex(ValueError,'mask changed'):
                finalize(p/'run',receipts,expected_collections=collections,required_stages=receipts)
            self.assertFalse((p/'run/CURRENT.json').exists())

    def test_scope_mismatch_and_partial_full_pipeline_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,_=self.fixture(p)
            for path in receipts.values():
                value=json.loads(path.read_text());value['semantics']['scope']='full'
                path.write_text(json.dumps(value))
            # Refresh only the dependency bindings; stage inputs themselves have not changed.
            for name,parent in [('records','normalize'),('source_view','records')]:
                value=json.loads(receipts[name].read_text());value['dependencies'][parent]=file_sha(receipts[parent])
                receipts[name].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'every production stage'):
                finalize(p/'run',receipts,expected_collections=collections,required_stages=receipts)
            self.assertFalse((p/'run/CURRENT.json').exists())

    def test_wrapper_cannot_promote_bounded_native_snapshot_to_full(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,_=self.fixture(p)
            native=p/'snapshot';native.mkdir()
            catalog=[{'collection':'dcvlm','path':'a.parquet'}, {'collection':'genund','path':'b.parquet'}]
            atomic_json(native/'catalog.json',catalog)
            for native_scope in ['bounded_prefix','full_declared_inputs']:
                manifest={'schema_version':'corpus.native.v2','collections':list(collections),
                          'scope':native_scope,'statistics':{'samples':3}}
                atomic_json(native/'manifest.json',manifest)
                complete={'schema_version':'corpus.native.v2','complete':True,'scope':native_scope,
                          'statistics':{'samples':3},'manifest_sha256':file_sha(native/'manifest.json'),
                          'catalog_sha256':file_sha(native/'catalog.json')}
                atomic_json(native/'COMPLETE.json',complete)
                collections={name:digest({'snapshot':file_sha(native/'COMPLETE.json'),'collection':name,
                                         'files':[item for item in catalog if item['collection']==name]})
                             for name in collections}
                for name in REQUIRED_STAGES:
                    if name not in receipts:
                        receipts[name]=p/(name+'.json')
                        seal_receipt(receipts[name],stage=name,collections=collections,
                            coverage={'unit':'assets','expected':1,'processed':1,'missing':0,'pending_infra':0},
                            artifacts=[p/'native.parquet'])
                    value=json.loads(receipts[name].read_text());value['semantics']['scope']='full'
                    value['collections']=collections
                    if name=='normalize':
                        value['artifacts']=[artifact(native/file) for file in ['COMPLETE.json','manifest.json','catalog.json']]
                    receipts[name].write_text(json.dumps(value))
                for name,parent in [('records','normalize'),('source_view','records')]:
                    value=json.loads(receipts[name].read_text());value['dependencies'][parent]=file_sha(receipts[parent])
                    receipts[name].write_text(json.dumps(value))
                if native_scope=='bounded_prefix':
                    with self.assertRaisesRegex(ValueError,'cannot publish as full'):
                        finalize(p/'run',receipts,expected_collections=collections)
                    self.assertFalse((p/'run/CURRENT.json').exists())
                else:
                    self.assertEqual(finalize(p/'run',receipts,expected_collections=collections)['scope'],'full')

    def test_infra_pending_and_stale_dependencies_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,_=self.fixture(p)
            with self.assertRaisesRegex(ValueError,'infrastructure'):
                seal_receipt(p/'bad.json',stage='screening',collections=collections,
                    coverage={'unit':'assets','expected':2,'processed':2,'missing':0,'pending_infra':1},artifacts=[p/'native.parquet'])
            value=json.loads(receipts['records'].read_text());value['dependencies']['normalize']='f'*64
            receipts['records'].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'dependency'):
                finalize(p/'run',receipts,expected_collections=collections,required_stages=receipts)

    def test_single_collection_full_publication_binds_explicit_scope_at_every_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);collections,receipts,rows=self.fixture(p,single_collection=True)
            native=p/'snapshot';native.mkdir()
            catalog=[{'collection':'dcvlm','path':'a.parquet'}]
            atomic_json(native/'catalog.json',catalog)
            atomic_json(native/'manifest.json',{'schema_version':'corpus.native.v2','collections':['dcvlm'],
                'scope':'full_declared_inputs','statistics':{'samples':3}})
            atomic_json(native/'COMPLETE.json',{'schema_version':'corpus.native.v2','complete':True,
                'scope':'full_declared_inputs','statistics':{'samples':3},
                'manifest_sha256':file_sha(native/'manifest.json'),'catalog_sha256':file_sha(native/'catalog.json')})
            collections={'dcvlm':digest({'snapshot':file_sha(native/'COMPLETE.json'),
                'collection':'dcvlm','files':catalog})}
            for name in REQUIRED_STAGES:
                if name not in receipts:
                    receipts[name]=p/(name+'.json')
                    seal_receipt(receipts[name],stage=name,collections=collections,
                        coverage={'unit':'assets','expected':1,'processed':1,'missing':0,'pending_infra':0},
                        artifacts=[p/'native.parquet'])
                value=json.loads(receipts[name].read_text())
                value['semantics'].update(scope='full',required_collections=['dcvlm'])
                value['collections']=collections
                if name=='normalize':
                    value['artifacts']=[artifact(native/file) for file in ['COMPLETE.json','manifest.json','catalog.json']]
                receipts[name].write_text(json.dumps(value))
            for name,parent in [('records','normalize'),('source_view','records')]:
                value=json.loads(receipts[name].read_text());value['dependencies'][parent]=file_sha(receipts[parent])
                receipts[name].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'snapshots differ from required_collections'):
                finalize(p/'run',receipts,expected_collections=collections)
            omitted=json.loads(receipts['frontpass'].read_text())
            declared=omitted['semantics'].pop('required_collections')
            receipts['frontpass'].write_text(json.dumps(omitted))
            with self.assertRaisesRegex(ValueError,'Stage required_collections'):
                finalize(p/'run',receipts,expected_collections=collections,required_collections=['dcvlm'])
            self.assertFalse((p/'run/CURRENT.json').exists())
            omitted['semantics']['required_collections']=declared
            receipts['frontpass'].write_text(json.dumps(omitted))
            result=finalize(p/'run',receipts,expected_collections=collections,required_collections=['dcvlm'])
            self.assertEqual(result['scope'],'full')
            self.assertEqual(result['required_collections'],['dcvlm'])
            manifest=read_manifest(p/'run')
            self.assertEqual(manifest['required_collections'],['dcvlm'])
            self.assertEqual(set(manifest['per_collection']),{'dcvlm'})
            self.assertEqual([row for batch in iter_batches(p/'run') for row in batch.to_pylist()],rows[:2])


if __name__=='__main__':unittest.main()
