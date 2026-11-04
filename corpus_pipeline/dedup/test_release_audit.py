"""Combined-index reverse joins and independently rechecked source identities."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from .asset_release import artifact
from .asset_similarity_index import build
from .release_audit import combined_audit


class CombinedAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = root = Path(self.temp.name)
        self.index = index = root / 'index'
        graph = root / 'graph'
        for directory in [index/'catalog', graph/'representatives', root/'assets']:
            directory.mkdir(parents=True)
        self.a, self.b, self.c = [f'{i:064x}' for i in (1, 2, 3)]
        assets = root/'assets/data.parquet'
        pq.write_table(pa.Table.from_pylist([
            {'asset_id':self.a,'canonical_asset_id':self.a,'screening_action':'retain_annotated'},
            {'asset_id':self.b,'canonical_asset_id':self.b,'screening_action':'retain_annotated'},
            {'asset_id':self.c,'canonical_asset_id':self.c,'screening_action':'exclude'},
        ]), assets)
        am = root/'assets/COMPLETE.json'
        am.write_text(json.dumps({'complete':True,'canonical_equivalence_only':True,
                                 'counts':{'assets':3},'files':[artifact(assets)]}))
        self.mapping = index/'mapping.parquet'
        pq.write_table(pa.table({'vector_id':[0,1],'asset_id':[self.a,self.b]}),self.mapping)
        item = artifact(self.mapping);item['path']='mapping.parquet'
        receipt=index/'block.json';receipt.write_text(json.dumps({'artifacts':[item]}))
        catalog=index/'catalog/COMPLETE.json'
        catalog.write_text(json.dumps({'blocks':[{'receipt_path':'block.json',
                                                'receipt_sha256':artifact(receipt)['sha256']}]}))
        self.membership=graph/'representatives/membership.parquet'
        pq.write_table(pa.table({'vector_id':[0,1],'component_id':[0,0],'component_size':[2,2],
            'candidate_rep_vector_id':[0,0],'direct_score':[1.0,0.99],
            'is_candidate_representative':[True,False],'eligible':[True,True]}), self.membership)
        item=artifact(self.membership);item['path']='representatives/membership.parquet'
        rep=graph/'representatives/COMPLETE.json';rep.write_text(json.dumps({'artifacts':[item]}))
        plan=graph/'PLAN.json';plan.write_text(json.dumps({'identity':{'catalog_sha256':artifact(catalog)['sha256']}}))
        (graph/'COMPLETE.json').write_text(json.dumps({'complete':True,'advisory':True,
            'automatic_drop':False,'source_artifact_hashes_reverified':True,'vectors':2,
            'plan_sha256':artifact(plan)['sha256'],'phases':{'representatives':artifact(rep)['sha256']}}))
        self.output=root/'asset_similarity_index'
        build(am,index,graph,self.output,threads=1,memory='1GB')

    def test_roundtrip_preserves_unmapped_and_excluded_assets(self):
        result=combined_audit(self.root)
        self.assertEqual((result['assets'],result['vectors'],result['mismatches']),(3,2,0))
        self.assertTrue(result['all_mapping_hashes_verified'])
        self.assertTrue(result['graph_membership_hash_verified'])

    def test_mapping_and_membership_content_reverified(self):
        original=self.mapping.read_bytes();self.mapping.write_bytes(original+b'changed')
        with self.assertRaises(ValueError):combined_audit(self.root)
        self.mapping.write_bytes(original)
        self.membership.write_bytes(self.membership.read_bytes()+b'changed')
        with self.assertRaises(ValueError):combined_audit(self.root)

    def test_hash_consistent_but_wrong_representative_rejected(self):
        path=self.output/'COMPLETE.json';complete=json.loads(path.read_text())
        for i,item in enumerate(complete['files']):
            target=Path(item['path']);table=pq.ParquetFile(target).read();rows=table.to_pylist()
            changed=False
            for row in rows:
                if row['asset_id']==self.b:
                    row['sscd_candidate_rep_asset_id']=self.b;changed=True
            if changed:
                pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),target)
                complete['files'][i]=artifact(target)
        path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError,'identity mapping mismatch'):
            combined_audit(self.root)


if __name__=='__main__':unittest.main()
