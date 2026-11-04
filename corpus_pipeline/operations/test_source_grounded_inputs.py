"""Evidence integrity and multi-image scope regression checks (no inference)."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .source_grounded_inputs import prepare_one, digest


class SourceEvidenceTests(unittest.TestCase):
    def fixture(self, directory, text=None):
        stream=io.BytesIO();Image.new('RGB',(11,7),(1,80,4)).save(stream,format='PNG');image=stream.getvalue()
        key='record';annotations={'conversations.txt':text or 'human:First <image> then second <image><EOCL>\ngpt:Two images.',
                                 'data_type.txt':'instruction','ds_name.txt':'source','orig_url.txt':'https://example.invalid/reference'}
        payload=bytearray();components=[]
        for member,raw in [(key+'.1.png',image),*((key+'.'+name,value.encode()) for name,value in annotations.items())]:
            components.append({'member':member,'offset_data':len(payload),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});payload.extend(raw)
        archive=directory/'source.tar';archive.write_bytes(payload);aid=hashlib.sha256(image).hexdigest()
        locator={'path':'source.tar','source_key':key,'archive_sha256':hashlib.sha256(payload).hexdigest(),'components':components}
        ref={'member':key+'.1.png','path':'source.tar','dataset_id':'fixture','archive_sha256':locator['archive_sha256'],
             'archive_size':len(payload),'offset_data':0,'size':len(image),'sha256':aid}
        seed={'item_id':aid,'ref':ref,'source':'source','image_index':1,'sample_id':'sample','occurrence_count':12,'source_count':3,
              'prior_screening':{},'reconciliation_binding':{},'reconciliation_receipt':{}}
        native={'source_key':key,'image_count':2,'original_locator_json':json.dumps(locator),
                'adapter_metadata_json':json.dumps({'pairing_state':'verified'})}
        return seed,native,archive,annotations

    def test_original_bytes_and_target_slot_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seed,native,archive,annotations=self.fixture(root)
            item=prepare_one(seed,native,{},archive,root/'output');record=item['source_records'][0]
            self.assertEqual(record['target_image_index_zero_based'],1)
            self.assertEqual(record['record_image_count'],2)
            self.assertTrue(record['placeholder_alignment_verified'])
            self.assertEqual(item['scope']['unread_occurrences'],11)
            self.assertEqual(item['scope']['other_sources_not_inspected'],2)
            self.assertFalse(item['scope']['all_original_record_images_decoded'])
            self.assertEqual(hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest(),seed['item_id'])
            raw=json.loads(Path(record['full_record_evidence']['path']).read_text())
            self.assertEqual(raw['annotations'],annotations)
            self.assertEqual(raw['annotation_content_sha256'],digest(annotations))

    def test_tampered_original_text_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seed,native,archive,_=self.fixture(root)
            raw=bytearray(archive.read_bytes());raw[-1]^=1;archive.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'component bytes changed'):
                prepare_one(seed,native,{},archive,root/'output')

    def test_unresolved_placeholder_mapping_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seed,native,archive,_=self.fixture(root,'human:<image> literal <image> extra <image><EOCL>\ngpt:Ambiguous.')
            item=prepare_one(seed,native,{},archive,root/'output')
            self.assertFalse(item['source_records'][0]['placeholder_alignment_verified'])
            self.assertIsNone(item['source_records'][0]['target_turn_zero_based'])

    def test_truncation_keeps_full_original_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);text='human:<image>'+('a'*30000)+'<image><EOCL>\ngpt:End.'
            seed,native,archive,_=self.fixture(root,text)
            item=prepare_one(seed,native,{},archive,root/'output');record=item['source_records'][0]
            self.assertTrue(item['scope']['text_truncated'])
            self.assertLessEqual(item['scope']['source_context_characters'],24000)
            raw=json.loads(Path(record['full_record_evidence']['path']).read_text())
            self.assertEqual(raw['annotations']['conversations.txt'],text)


if __name__=='__main__':unittest.main()
