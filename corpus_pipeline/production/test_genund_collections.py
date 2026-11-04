import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from . import genund_collections as g
from .portable_refs import read_ref_bytes

DATASET = 'inspire/gen-und-mm-pretrain@'+'a'*64


def write_tar(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'w') as stream:
        for name,data in values:
            data = data.encode() if isinstance(data,str) else data
            info=tarfile.TarInfo(name); info.size=len(data); stream.addfile(info,io.BytesIO(data))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name); self.root=self.base/'raw'; self.root.mkdir()

    def test_streamed_json_requires_complete_array_and_preserves_unicode(self):
        path=self.base/'rows.json'; path.write_text(json.dumps([{'s':'原始\n文本'},{'s':'x'}]))
        self.assertEqual(list(g.iter_json_records(path)),[(0,{'s':'原始\n文本'}),(1,{'s':'x'})])
        for broken in ('[{"x":1}', '[{"x":1},]', '[{"x":1}] garbage'):
            path.write_text(broken)
            with self.assertRaises(ValueError): list(g.iter_json_records(path))

    def test_scalecap_dual_views_share_asset_and_preserve_original_conversation(self):
        path=self.root/'ScaleCap450k_image/images/share4v_image/a.jpg'; path.parent.mkdir(parents=True); path.write_bytes(b'image-a')
        annotation=self.base/'ScaleCap450k.json'; original={'images':['images/share4v_image/a.jpg'],
            'conversations':[{'from':'human','value':'<image>\nDescribe exactly.'},{'from':'gpt','value':' Original caption\nunchanged. '}]}
        annotation.write_text(json.dumps([original]))
        spec={'adapter':'genund_scalecap','root':str(self.root),'dataset_id':DATASET,'annotation_paths':[str(annotation)]}
        units=g.enumerate_units(spec,self.base/'stage'); rows=list(g.iter_records(units[0]))
        self.assertEqual(len(rows),2); i2t,t2i=rows
        self.assertEqual(i2t['source_image_refs'],t2i['source_image_refs'])
        self.assertIs(i2t['raw']['image_loss_allowed'],False); self.assertIs(t2i['raw']['image_loss_allowed'],True)
        self.assertEqual(i2t['raw']['source_annotations'],original)
        self.assertEqual(t2i['raw']['segments'][0]['text'],' Original caption\nunchanged. ')
        moved=self.base/'moved'; shutil.copytree(self.root,moved)
        self.assertEqual(read_ref_bytes(i2t['source_image_refs'][0],{DATASET:str(moved)}),b'image-a')
        self.assertEqual(g.enumerate_units(spec,self.base/'stage'),units)
        annotation.write_text('[]')
        with self.assertRaisesRegex(ValueError,'source annotation changed'): list(g.iter_records(units[0]))

    def test_finevision_keeps_multi_image_multi_turn_and_checks_explicit_binding(self):
        refs=[]
        for i in range(2):
            path=self.root/f'FineVision/source/{i}.jpg'; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(f'image-{i}'.encode())
            refs.append({'path':f'source/{i}.jpg','sha256':g.file_sha(path)})
        row={'images':refs,'texts':[{'user':'Compare these.','assistant':'They differ.'},
                                  {'user':'Where exactly?','assistant':'On the left.'}], 'source':'source', 'formatting_min':3}
        unit={'source_shard':'FineVision/source/annotations.json','subset':'source'}
        with g.Storage(self.root,DATASET) as store: result=list(g.finevision_record(unit,row,4,store))[0]
        parts=result['raw']['segments']; self.assertEqual([p['image_index'] for p in parts if p['kind']=='image'],[0,1])
        self.assertEqual([p['text'] for p in parts if p['kind']=='text'],['Compare these.','They differ.','Where exactly?','On the left.'])
        self.assertEqual([p['turn'] for p in parts if p['kind']=='text'],[0,1,2,3]); self.assertEqual(result['metadata']['issues'],[])
        refs[1]['sha256']='0'*64
        with g.Storage(self.root,DATASET) as store:
            with self.assertRaisesRegex(ValueError,'SHA differs'): list(g.finevision_record(unit,row,4,store))

    def test_openuni_exact_shard_join_retains_captions_and_ignores_url_default(self):
        prefix='openuni/data/megalith-10m'; raw=self.root/prefix/'raw/00001'; raw.mkdir(parents=True); (raw/'000010003.jpg').write_bytes(b'image')
        annotation={'key':'000010003','caption':'https://flickr.com/photo','caption_internlm2':'Long caption',
                    'caption_internlm2_short':'Short caption','caption_florence2':'Long caption'}
        write_tar(self.root/prefix/'captions/00001.tar',[('./00001/000010003.json',json.dumps(annotation))])
        unit={'source_shard':prefix+'/megalith10m_all.json','subset':'megalith-10m','image_root':prefix+'/raw','annotation_root':prefix+'/captions'}
        with g.Storage(self.root,DATASET) as store:
            rows=list(g.openuni_record(unit,{'image':'00001/000010003.jpg','annotation':'00001/000010003.json'},0,store))
        self.assertEqual([r['raw']['segments'][0]['text'] for r in rows],['Long caption','Short caption'])
        self.assertEqual(rows[0]['source_image_refs'],rows[1]['source_image_refs'])
        self.assertTrue(all(r['raw']['direction']=='t2i' for r in rows))

    def test_openuni_tar_portable_and_generator_provenance_is_not_caption_keyword(self):
        prefix='openuni/data/text-to-image-2M/raw/data_1024_10K'
        write_tar(self.root/prefix/'data_000000.tar',[
            ('flux_1024_10k_00000000.jpg',b'image'),
            ('flux_1024_10k_00000000.json',json.dumps({'prompt':'A landscape','image_origin':'natural'}))])
        unit={'source_shard':'manifest.json','subset':'text-to-image-2M','image_root':prefix,'annotation_root':prefix}
        record={'image':'data_000000/flux_1024_10k_00000000.jpg','annotation':'data_000000/flux_1024_10k_00000000.json'}
        with g.Storage(self.root,DATASET) as store: row=list(g.openuni_record(unit,record,0,store))[0]
        self.assertEqual(row['raw']['image_origin'],'ai_generated')
        moved=self.base/'moved'; shutil.copytree(self.root,moved)
        self.assertEqual(read_ref_bytes(row['source_image_refs'][0],{DATASET:str(moved)}),b'image')
        with g.Storage(self.root,DATASET) as store:
            with self.assertRaises(ValueError): store.locate(prefix,'../elsewhere.jpg')

    def test_missing_bindings_are_kept_as_quarantined_original_records(self):
        row={'images':['images/share4v_image/missing.jpg'],'conversations':[{'from':'human','value':'<image>Question'},{'from':'gpt','value':'Answer'}]}
        with g.Storage(self.root,DATASET) as store:
            rows=list(g.scalecap_record({'source_shard':'ScaleCap450k.json'},row,0,store))
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['metadata']['prepared_adapter_status']=='quarantine' for r in rows))
        self.assertEqual(rows[0]['raw']['segments'][0]['text'],'<image>Question')
        self.assertEqual(rows[0]['raw']['source_annotations'],row)

    def test_scalecap_empty_source_is_quarantined_and_verified_zip_recovery_is_portable(self):
        source=self.root/'ScaleCap450k_image/images/share4v_image/broken.jpg'; source.parent.mkdir(parents=True); source.write_bytes(b'')
        row={'images':['share4v_image/broken.jpg'],'conversations':[{'from':'human','value':'<image>'},{'from':'gpt','value':'Original.'}]}
        unit={'source_shard':'ScaleCap450k_image/ScaleCap450k.json'}
        with g.Storage(self.root,DATASET) as store: failed=list(g.scalecap_record(unit,row,2,store))
        self.assertTrue(all('corrupt_image' in item['metadata']['issues'] for item in failed))
        recovered=self.base/'recovered'; path=recovered/'ScaleCap/a.img'; path.parent.mkdir(parents=True); path.write_bytes(b'original-zip-image')
        dataset='inspire/scalecap-recovered@'+'b'*64
        mapping=self.base/'binding.json'; mapping.write_text(json.dumps({'complete':True,'binding':{'dataset_id':dataset},
            'images':{'ScaleCap450k_image/images/share4v_image/broken.jpg':{'path':'ScaleCap/a.img',
                'dataset_id':dataset,'sha256':g.file_sha(path),'official_zip_lfs_sha_verified':True}}}))
        unit.update(image_recovery_map=str(mapping),image_recovery_map_sha256=g.file_sha(mapping),
                    recovered_root=str(recovered),recovered_dataset_id=dataset)
        with g.Storage(self.root,DATASET) as store: rows=list(g.scalecap_record(unit,row,2,store))
        self.assertEqual(rows[0]['metadata']['issues'],[])
        self.assertEqual(rows[0]['raw']['source_annotations'],row)
        self.assertEqual(read_ref_bytes(rows[0]['source_image_refs'][0],{dataset:str(recovered)}),b'original-zip-image')
        self.assertEqual(source.read_bytes(),b'')


if __name__=='__main__': unittest.main()
