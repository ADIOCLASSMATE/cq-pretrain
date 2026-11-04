import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from .asset_release import artifact, build


class AssetReleaseTest(unittest.TestCase):
    def test_screening_does_not_propagate_and_conflicts_do_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); recon=root/'recon';recon.mkdir()
            def row(i, **extra):
                return dict(asset_id=f'{i:064x}',pixel_sha256='a'*64,pixel_hash_count=1,
                    pixel_sha256_conflict=False,dimension_count=1,has_invalid_observation=False,
                    original_width=256,original_height=256,occurrence_count=1,source='test',
                    sscd_usable=True,global_status='no_detector_alert',**extra)
            rows=[row(1),row(2),row(3),row(4)]
            rows[2]['pixel_sha256_conflict']=True
            rows[3]['global_status']='rejected_too_small'
            rows[3]['pixel_sha256']='b'*64
            data=recon/'assets.parquet';pq.write_table(pa.Table.from_pylist(rows),data)
            art=artifact(data);art['path']='assets.parquet'
            receipt=recon/'bucket.json';receipt.write_text(json.dumps({'artifacts':[art]}))
            (recon/'COMPLETE.json').write_text(json.dumps({'complete':True,'counts':{'assets':4},
                'buckets':[{'receipt_path':'bucket.json','receipt_sha256':artifact(receipt)['sha256']}]}))
            screen=root/'screen.parquet'
            pq.write_table(pa.Table.from_pylist([{'asset_id':f'{1:064x}','screen_action':'exclude',
                'screen_reason':'confirmed_ai','quality_reviewed':False,'strict_blur_reviewed':False}]),screen)
            manifest=root/'screen.json';manifest.write_text(json.dumps({'complete':True,'files':[artifact(screen)]}))
            out=root/'out';done=build(recon,manifest,out,threads=1,memory='1GB')
            all_rows={r['asset_id']:r for f in done['files'] for r in pq.read_table(f['path']).to_pylist()}
            a,b,c,d=[all_rows[f'{i:064x}'] for i in range(1,5)]
            self.assertEqual(a['canonical_asset_id'],b['asset_id'])
            self.assertEqual(b['screening_action'],'retain_annotated')
            self.assertEqual(a['screening_action'],'exclude')
            self.assertEqual(c['canonical_asset_id'],c['asset_id'])
            self.assertEqual(d['screening_action'],'exclude')
            self.assertTrue(done['canonical_equivalence_only'])
            self.assertEqual(done['counts']['encoded_assets_sharing_exact_pixels'],1)
            self.assertEqual(build(recon,manifest,out,threads=1,memory='1GB'),done)
            Path(done['files'][0]['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                build(recon,manifest,out,threads=1,memory='1GB')


if __name__=='__main__': unittest.main()
