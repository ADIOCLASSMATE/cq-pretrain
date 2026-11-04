import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from corpus_pipeline.production.common import atomic_json, canonical, file_sha
from .pixmo_recovery import CONTRACT, stamp
from .pixmo_retry import retry, finalize


class PixmoRetryTests(unittest.TestCase):
    def test_retry_requires_official_hash_and_preserves_exact_original_bytes(self):
        blob=io.BytesIO();Image.new('RGB',(9,9),'red').save(blob,format='PNG');blob=blob.getvalue();sha=hashlib.sha256(blob).hexdigest()
        with tempfile.TemporaryDirectory() as tmp, patch('corpus_pipeline.unified.pixmo_retry.fetch',return_value=blob):
            root=Path(tmp)/'assets';dataset='inspire/pixmo@'+'a'*64
            rejected=retry(('f'*64,['https://example.test/image']),root,dataset)
            self.assertEqual(rejected['status'],'unbound');self.assertFalse(root.exists())
            result=retry((sha,['https://example.test/image']),root,dataset)
            self.assertEqual(result['status'],'bound');self.assertEqual((root/result['relative_path']).read_bytes(),blob)

    def test_final_binding_waits_for_retry_and_does_not_rewrite_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);primary=base/'primary.sqlite';db=sqlite3.connect(primary)
            db.execute('CREATE TABLE bindings(original_sha256 TEXT PRIMARY KEY,status TEXT,result_json TEXT)')
            db.execute('INSERT INTO bindings VALUES(?,?,?)',('a'*64,'unbound',canonical({'original_sha256':'a'*64,'status':'unbound'})));db.commit();db.close()
            before=file_sha(primary);receipt={'contract':CONTRACT,'path':str(primary),'sha256':before,'fingerprint':stamp(primary),'counts':{'unbound':1}}
            retried=base/'retry';worker=retried/'worker-0';worker.mkdir(parents=True)
            with patch('corpus_pipeline.unified.pixmo_retry.finalize_bindings',return_value=receipt):
                self.assertIsNone(finalize(base/'original',retried,base/'final',1))
                database=worker/'results.sqlite';db=sqlite3.connect(database);db.execute('CREATE TABLE results(original_sha256 TEXT,status TEXT,result_json TEXT)')
                value={'original_sha256':'a'*64,'status':'bound','local_sha256':'a'*64,'proof':{'source_sha256':'a'*64}}
                db.execute('INSERT INTO results VALUES(?,?,?)',('a'*64,'bound',canonical(value)));db.commit();db.close()
                atomic_json(worker/'COMPLETE.json',{'path':str(database),'sha256':file_sha(database)})
                result=finalize(base/'original',retried,base/'final',1)
            self.assertEqual(result['counts'],{'bound':1});self.assertEqual(result['retry_added_bound_images'],1)
            self.assertEqual(file_sha(primary),before)


if __name__=='__main__':unittest.main()
