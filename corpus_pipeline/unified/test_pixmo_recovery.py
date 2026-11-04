import hashlib,io,json,sqlite3,unittest
from PIL import Image
from .pixmo_recovery import describe,verify_blob

class PixmoRecoveryTests(unittest.TestCase):
    def fixture(self):
        source=Image.new('RGB',(160,128),'red');raw=io.BytesIO();source.save(raw,format='PNG')
        jpeg=io.BytesIO();source.save(jpeg,format='JPEG',quality=75)
        original,target=raw.getvalue(),jpeg.getvalue();info=describe(target)
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE images(local_sha256 TEXT,pixel_sha256 TEXT,relative_path TEXT,width INT,height INT,profile_json TEXT)')
        db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)',(info['local_sha256'],info['pixel_sha256'],'pixmo-ask-model-anything/0001/0.jpg',info['width'],info['height'],info['profile_json']))
        return original,target,db,[json.loads(info['profile_json'])]
    def test_exact_transcode_proves_binding_without_filename_guess(self):
        original,target,db,profiles=self.fixture()
        result=verify_blob(original,hashlib.sha256(original).hexdigest(),db,profiles)
        self.assertEqual(result['status'],'bound');self.assertEqual(result['local_sha256'],hashlib.sha256(target).hexdigest())
        self.assertEqual(result['proof']['method'],'exact_reencoded_sha256')
    def test_changed_original_is_never_bound_even_if_its_pixels_match(self):
        original,target,db,profiles=self.fixture()
        self.assertEqual(verify_blob(original,'a'*64,db,profiles)['status'],'source_changed')
    def test_different_image_is_not_an_approximate_match(self):
        original,target,db,profiles=self.fixture();stream=io.BytesIO();Image.new('RGB',(160,128),'blue').save(stream,format='PNG');blob=stream.getvalue()
        self.assertEqual(verify_blob(blob,hashlib.sha256(blob).hexdigest(),db,profiles)['status'],'unbound')

if __name__=='__main__':unittest.main()
