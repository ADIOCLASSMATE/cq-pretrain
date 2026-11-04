import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from .asset_release import artifact
from .asset_similarity_index import build


class SimilarityIndexTest(unittest.TestCase):
    def test_similar_members_do_not_replace_exact_identity_or_screening(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);index=root/'index';graph=root/'graph'
            (index/'catalog').mkdir(parents=True);(graph/'representatives').mkdir(parents=True)
            a,b,c=[f'{i:064x}' for i in (1,2,3)]
            assets=root/'assets.parquet'
            pq.write_table(pa.Table.from_pylist([
                {'asset_id':a,'canonical_asset_id':a,'screening_action':'retain_annotated'},
                {'asset_id':b,'canonical_asset_id':b,'screening_action':'retain_annotated'},
                {'asset_id':c,'canonical_asset_id':c,'screening_action':'exclude'}]),assets)
            am=root/'assets.json';am.write_text(json.dumps({'complete':True,'canonical_equivalence_only':True,'counts':{'assets':3},'files':[artifact(assets)]}))
            mapping=index/'mapping.parquet'
            pq.write_table(pa.table({'vector_id':pa.array([0,1],type=pa.int64()),'asset_id':[a,b]}),mapping)
            r=index/'block.json';item=artifact(mapping);item['path']='mapping.parquet'
            r.write_text(json.dumps({'artifacts':[item]}))
            cat=index/'catalog/COMPLETE.json'
            cat.write_text(json.dumps({'blocks':[{'receipt_path':'block.json','receipt_sha256':artifact(r)['sha256']}]}))
            membership=graph/'representatives/membership.parquet'
            pq.write_table(pa.table({'vector_id':[0,1],'component_id':[0,0],'component_size':[2,2],
                'candidate_rep_vector_id':[0,0],'direct_score':[1.0,0.99],
                'is_candidate_representative':[True,False],'eligible':[True,True]}),membership)
            rep=graph/'representatives/COMPLETE.json';item=artifact(membership);item['path']='representatives/membership.parquet'
            rep.write_text(json.dumps({'artifacts':[item]}))
            plan=graph/'PLAN.json';plan.write_text(json.dumps({'identity':{'catalog_sha256':artifact(cat)['sha256']}}))
            complete=graph/'COMPLETE.json';complete.write_text(json.dumps({'complete':True,'advisory':True,
                'automatic_drop':False,'vectors':2,'plan_sha256':artifact(plan)['sha256'],
                'phases':{'representatives':artifact(rep)['sha256']}}))
            result=build(am,index,graph,root/'output',threads=1,memory='1GB')
            rows={row['asset_id']:row for item in result['files'] for row in pq.read_table(item['path']).to_pylist()}
            self.assertEqual(rows[b]['sscd_candidate_rep_asset_id'],a)
            self.assertEqual(rows[b]['canonical_asset_id'],b)
            self.assertIsNone(rows[c]['sscd_component_id'])
            self.assertEqual(rows[c]['screening_action'],'exclude')
            self.assertEqual(result['assets'],3)
            self.assertTrue(all(row['sscd_candidate_only'] for row in rows.values()))
            unsafe=json.loads(am.read_text());unsafe['canonical_equivalence_only']=False
            unsafe_path=root/'unsafe.json';unsafe_path.write_text(json.dumps(unsafe))
            with self.assertRaisesRegex(ValueError,'exact equivalences'):
                build(unsafe_path,index,graph,root/'unsafe_output',threads=1,memory='1GB')
            cat.write_text(cat.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'catalogs differ'):
                build(am,index,graph,root/'changed',threads=1,memory='1GB')


if __name__=='__main__':unittest.main()
