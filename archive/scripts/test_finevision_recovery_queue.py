import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import finevision_recovery_queue as recovery
import recover_genund_annotation_ranges as ranges
from finevision_local_assets import build_subset
from corpus_pipeline.production.common import atomic_json
from corpus_pipeline.production.genund_collections import Storage,finevision_record,iter_records,stamp
from corpus_pipeline.production.portable_refs import read_ref_bytes


class RecoveryTest(unittest.TestCase):
    def fixture(self, directory):
        rows=[{'images':[{'bytes':b'first-image','path':None},{'bytes':b'second-image','path':None}],
               'texts':[{'user':'Compare both.','assistant':'They differ.'}]},
              {'images':[{'bytes':b'third-image','path':None}],
               'texts':[{'user':'Describe.','assistant':'A third image.'}]}]
        schema=pa.schema([('images',pa.list_(pa.struct([('bytes',pa.binary()),('path',pa.string())]))),
                          ('texts',pa.list_(pa.struct([('user',pa.string()),('assistant',pa.string())])))])
        stream=io.BytesIO(); pq.write_table(pa.Table.from_pylist(rows,schema=schema),stream,row_group_size=1)
        blob=stream.getvalue(); root=Path(directory)/'mount'; leaf=root/'FineVision/example'; leaf.mkdir(parents=True)
        for name,data in [('9.jpg',b'first-image'),('10.jpg',b'second-image'),('11.jpg',b'third-image')]:
            (leaf/name).write_bytes(data)
        task={'task_id':'test','subset':'example','path':'example/train-00001-of-00002.parquet',
              'revision':'revision','size':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
              'row_start':5,'image_start':9,'rows':2,'images':3}
        return blob,root,task

    def test_all_multiimage_bytes_bind_and_keep_ordered_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            blob,root,task=self.fixture(temporary); area=Path(temporary)/'run'
            with patch.object(recovery,'fetch',side_effect=lambda url,start,stop:blob[start:stop]):
                result=recovery.recover(task,{'root':str(root)},area)
            rows=[json.loads(line) for line in Path(result['path']).read_text().splitlines()]
            self.assertEqual([image['path'] for image in rows[0]['images']],
                             ['FineVision/example/9.jpg','FineVision/example/10.jpg'])
            self.assertEqual(rows[0]['texts'],[{'user':'Compare both.','assistant':'They differ.'}])
            self.assertEqual(result['image_slots'],3)
            self.assertNotIn('bytes',rows[0]['images'][0])
            (root/'FineVision/example/10.jpg').write_bytes(b'incorrect-but-same-name')
            with patch.object(recovery,'fetch',side_effect=lambda url,start,stop:blob[start:stop]):
                with self.assertRaisesRegex(ValueError,'lack verified mounted binding'):
                    recovery.recover(task,{'root':str(root)},Path(temporary)/'second')
            self.assertFalse((Path(temporary)/'second/recovered/test/COMPLETE.json').exists())

    def test_projected_null_paths_preserve_full_image_cardinality(self):
        with tempfile.TemporaryDirectory() as temporary:
            blob,root,task=self.fixture(temporary)
            fake=lambda url,start,stop:blob[start:stop]
            with patch.object(recovery,'fetch',side_effect=fake),patch.object(ranges,'fetch',side_effect=fake):
                result=recovery.inventory(task,Path(temporary)/'run')
            self.assertEqual((result['rows'],result['images']),(2,3))

    def test_missing_original_is_recovered_only_after_complete_local_sha_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            blob,root,task=self.fixture(temporary); area=Path(temporary)/'run'
            (root/'FineVision/example/10.jpg').unlink()
            # A reordered source image must bind by its bytes, not its row number.
            (root/'FineVision/example/9.jpg').rename(root/'FineVision/example/42.jpg')
            build_subset({'root':str(root)},area,'example',workers=2)
            restored=Path(temporary)/'recovered'; dataset='inspire/finevision-recovered@'+'f'*64
            atomic_json(area/'RECOVERED_ASSET_BINDING.json',{'root':str(restored),'dataset_id':dataset})
            with patch.object(recovery,'fetch',side_effect=lambda url,start,stop:blob[start:stop]):
                result=recovery.recover(task,{'root':str(root)},area)
            row=json.loads(Path(result['path']).read_text().splitlines()[0])
            self.assertEqual(row['images'][0]['path'],'FineVision/example/42.jpg')
            self.assertEqual(row['images'][1]['dataset_id'],dataset)
            self.assertEqual(result['recovered_missing_image_slots'],1)
            unit={'source_shard':task['path'],'recovered_root':str(restored),'recovered_dataset_id':dataset}
            original_dataset='inspire/genund@'+'a'*64
            with Storage(root,original_dataset) as store: record=next(finevision_record(unit,row,0,store))
            self.assertEqual(record['metadata']['issues'],[])
            second=record['source_image_refs'][1]
            self.assertEqual(read_ref_bytes(second,{dataset:str(restored)}),b'second-image')
            unit.update(adapter='genund_finevision',root=str(root),dataset_id=original_dataset,
                        path=result['path'],annotation_path=result['path'],fingerprint=stamp(result['path']),
                        annotation_fingerprint=stamp(result['path']),record_layout='original_jsonl')
            all_records=list(iter_records(unit))
            self.assertEqual(len(all_records),2)
            self.assertEqual(len(all_records[0]['source_image_refs']),2)
            self.assertTrue(all(not value['metadata']['issues'] for value in all_records))


if __name__=='__main__': unittest.main()
