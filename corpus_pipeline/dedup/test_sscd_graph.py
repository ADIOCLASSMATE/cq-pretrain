import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import sscd_graph as g
from ..postprocess import sscd_index as s


def fixture(root,n,edges,counts=None):
    counts=counts or [n//2,n-n//2];start=0;blocks=[]
    for count in counts:blocks.append({'start_id':start,'count':count});start+=count
    assert start==n
    plan={'vectors':n,'blocks':blocks,'identity':{'scope':'bounded_validation','threshold':.75,
          'catalog_sha256':'f'*64,'index_root':str(root/'unused_catalog')}}
    root.mkdir(parents=True,exist_ok=True);s.atomic_json(root/'PLAN.json',plan)
    digest=hashlib.sha256();task_number=0;total=0
    for i,a in enumerate(blocks):
        for j in range(i,len(blocks)):
            b=blocks[j];task={'number':task_number,'left':i,'right':j,
                'pairs':a['count']*b['count'] if i!=j else a['count']*(a['count']-1)//2}
            rows=[{'vector_id_a':x,'vector_id_b':y,'exact_cosine':score} for x,y,score in edges
                  if a['start_id']<=x<a['start_id']+a['count'] and b['start_id']<=y<b['start_id']+b['count']]
            directory=root/'tasks'/f'{i:06d}'/f'{j:06d}';directory.mkdir(parents=True)
            schema=pa.schema([('vector_id_a',pa.int64()),('vector_id_b',pa.int64()),('exact_cosine',pa.float32())])
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),directory/'edges.parquet')
            value={'complete':True,'binding':{'plan_sha256':s.file_sha(root/'PLAN.json'),'task':task},
                   'pairs_covered':task['pairs'],'edges':len(rows),'artifacts':[s.artifact(directory/'edges.parquet',root)]}
            s.atomic_json(directory/'COMPLETE.json',value);digest.update(s.file_sha(directory/'COMPLETE.json').encode())
            task_number+=1;total+=len(rows)
    assert total==len(edges)
    s.atomic_json(root/'COMPLETE.json',{'complete':True,'all_pairs_covered':True,'plan_sha256':s.file_sha(root/'PLAN.json'),
        'vectors':n,'candidate_edges':total,'total_tasks':task_number,'total_pairs':n*(n-1)//2,
        'task_receipts_sha256':digest.hexdigest()})


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source,self.output=self.root/'source',self.root/'graph'
        self.edges=[(0,1,.91),(1,2,.92),(3,4,.97)]
        fixture(self.source,7,self.edges,[2,3,2])

    def tearDown(self):self.tmp.cleanup()

    def prepare(self,threshold=.75):return g.prepare(self.source,self.output,threshold=threshold,allow_bounded=True)
    def structure(self):
        self.prepare();g.pack(self.output,workers=2);return g.structure(self.output)
    def mapping(self):return pq.read_table(self.output/'representatives/membership.parquet').to_pylist()

    def policy(self,name,values,dtype):
        path=self.root/name;np.asarray(values,dtype=dtype).tofile(path);return path

    def test_components_audit_and_nontransitive_direct_representatives(self):
        result=self.structure();self.assertEqual(result['components']['components'],4)
        self.assertEqual(result['adjacency']['memory_mode'],'ram_then_sequential_write')
        self.assertTrue(result['adjacency']['full_cursor_coverage_verified'])
        comp=np.fromfile(self.output/'components/component_id.u4',dtype='<u4')
        np.testing.assert_array_equal(comp,[0,0,0,3,3,5,6])
        sizes=np.fromfile(self.output/'components/component_size.u4',dtype='<u4')
        np.testing.assert_array_equal(sizes,[3,3,3,2,2,1,1])
        reps=g.representatives(self.output);rows=self.mapping()
        self.assertEqual(reps['memory_mode'],'ram_then_sequential_write')
        self.assertTrue(reps['full_representative_root_invariant_verified'])
        self.assertEqual([r['candidate_rep_vector_id'] for r in rows],[0,0,2,3,3,5,6])
        self.assertTrue(rows[2]['is_candidate_representative'])
        self.assertTrue(reps['advisory']);self.assertEqual(reps['retained_eligible_assets'],7)
        self.assertEqual(reps['supervision_records_dropped'],0)
        complete=g.finalize(self.output);self.assertEqual(complete,g.finalize(self.output))
        self.assertFalse(complete['automatic_drop']);self.assertFalse(complete['near_duplicates_resolved'])

    def test_quality_order_priority_descending_and_id_ties(self):
        self.structure();priority=self.policy('priority',[0,20,0,5,5,0,0],'<u8')
        g.representatives(self.output,priority=priority)
        rows=self.mapping();self.assertEqual([r['candidate_rep_vector_id'] for r in rows],[1,1,1,3,3,5,6])
        pairs={(a,b):score for a,b,score in self.edges}
        for row in rows:
            a,b=row['vector_id'],row['candidate_rep_vector_id']
            if a!=b:self.assertAlmostEqual(row['direct_score'],pairs[tuple(sorted((a,b)))],places=6)

    def test_protected_and_ineligible_do_not_suppress_others(self):
        self.structure();protected=self.policy('protected',[0,1,0,0,0,0,0],'u1')
        eligible=self.policy('eligible',[1,1,1,0,1,1,1],'u1')
        g.representatives(self.output,protected=protected,eligible=eligible)
        self.assertEqual([r['candidate_rep_vector_id'] for r in self.mapping()],[0,1,2,None,4,5,6])
        g.finalize(self.output)

    def test_projection_checkpoint_partial_resume_and_concurrent_workers(self):
        self.prepare();first=g.pack(self.output,workers=2,max_segments=1)
        self.assertEqual(first['new_segments'],1)
        with self.assertRaises(FileNotFoundError):g.structure(self.output)
        partial=self.output/'packed/.000001.crashed.tmp';partial.mkdir();(partial/'edges.bin').write_bytes(b'partial')
        self.assertEqual(g.pack(self.output,workers=3)['new_segments'],2)
        self.assertFalse(partial.exists());self.assertEqual(g.pack(self.output,workers=2)['new_segments'],0)
        a=g.structure(self.output);b=g.structure(self.output);self.assertEqual(a,b)

    def test_original_and_projected_payload_corruption_are_rejected(self):
        self.prepare();g.pack(self.output,workers=1)
        packed=self.output/'packed/000000/edges.bin';data=bytearray(packed.read_bytes());data[0]^=1;packed.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'SHA'):g.structure(self.output)

    def test_full_source_payload_is_verified_before_projection(self):
        self.prepare();path=self.source/'tasks/000000/000000/edges.parquet'
        data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'SHA'):g.pack(self.output,workers=1)
        self.assertFalse((self.output/'packed/000000/COMPLETE.json').exists())

    def test_policy_changes_and_missing_graph_payload_block_completion(self):
        self.structure();priority=self.policy('priority',range(7),'<u8')
        g.representatives(self.output,priority=priority)
        data=bytearray(priority.read_bytes());data[0]^=1;priority.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'policy'):g.finalize(self.output)
        (self.output/'adjacency/neighbors.u4').unlink()
        with self.assertRaisesRegex(ValueError,'missing'):g.finalize(self.output)

    def test_threshold_cannot_claim_edges_below_retrieval_and_filters_explicitly(self):
        with self.assertRaisesRegex(ValueError,'threshold'):self.prepare(.74)
        self.prepare(.95);g.pack(self.output,workers=1);g.structure(self.output);g.representatives(self.output)
        rows=self.mapping();self.assertEqual([r['candidate_rep_vector_id'] for r in rows],[0,1,2,3,3,5,6])
        self.assertEqual(g.finalize(self.output)['edges'],1)

    def test_empty_graph_isolated_vertices_and_zero_length_csr(self):
        source,output=self.root/'empty-source',self.root/'empty-output';fixture(source,3,[],[1,2])
        g.prepare(source,output,allow_bounded=True);g.pack(output,workers=2);g.structure(output);g.representatives(output)
        rows=pq.read_table(output/'representatives/membership.parquet').to_pylist()
        self.assertEqual([r['candidate_rep_vector_id'] for r in rows],[0,1,2])
        self.assertEqual(g.finalize(output)['edges'],0)

    def test_native_matches_python_connected_components_random_graphs(self):
        self.output.mkdir();native=g.Native(self.output);rng=np.random.default_rng(99)
        for n in (2,9,31):
            rows=[(a,b,float(rng.uniform(.75,1))) for a in range(n) for b in range(a+1,n) if rng.random()<.16]
            edges=np.array(rows,dtype=g.EDGE_DTYPE);parent=np.empty(n,np.uint32);sizes=np.empty(n,np.uint32);degree=np.empty(n,np.uint64)
            native.call('graph_initialize',n,parent,sizes,degree);native.call('graph_count',n,len(edges),edges,parent,sizes,degree,.75)
            comp=np.empty(n,np.uint32);count=np.empty(n,np.uint32);offset=np.empty(n+1,np.uint64)
            summary=np.zeros(6,np.uint64);hist=np.zeros(32,np.uint64)
            native.call('graph_components',n,parent,sizes,degree,comp,count,offset,summary,hist)
            groups=[{i} for i in range(n)]
            for a,b,_ in rows:
                union=groups[a]|groups[b]
                for x in union:groups[x]=union
            np.testing.assert_array_equal(comp,[min(x) for x in groups]);np.testing.assert_array_equal(count,list(map(len,groups)))
            self.assertEqual(int(offset[-1]),2*len(rows))


if __name__=='__main__':unittest.main()
