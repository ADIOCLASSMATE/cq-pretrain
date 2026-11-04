"""Schema fixtures plus immutable input/output and original-supervision checks."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from corpus_pipeline.dedup.training_release import text_fingerprint
from corpus_pipeline.production.common import digest, file_sha
from corpus_pipeline.production.ingest import build_shard_manifest, iter_records
from corpus_pipeline.production.records import snapshot, read_asset_bytes, load_original_record
from .ingest import build, enumerate_inputs
from unittest.mock import patch


def image_bytes(color='green'):
    stream = io.BytesIO()
    Image.new('RGB', (160, 128), color).save(stream, 'PNG')
    return stream.getvalue()


def write_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w') as archive:
        for name, value in members:
            blob = value.encode() if isinstance(value, str) else value
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
    return path


def balanced(root, examples=None):
    examples = examples or [('a', 'captioning', 'pixmo_cap', 'human:<image>\nPlease provide a descriptive caption for this image.<EOCL>\ngpt:A green square.', 1),
        ('b', 'instruction', 'vqa', 'human:<image>How many?<EOCL>\ngpt:One.<EOCL>\nhuman:Color?<EOCL>\ngpt:Green.', 1),
        ('c', 'mmdoc_mt', 'document', 'human:intro<image>middle<image>end<EOCL>\ngpt:Document text.', 2),
        ('d', 'text', 'text_source', 'human:2+2?<EOCL>\ngpt:4', 0)]
    members = []
    for key, kind, source, text, count in examples:
        members.extend((key + '.' + suffix, value) for suffix, value in
                       [('conversations.txt', text), ('data_type.txt', kind), ('ds_name.txt', source),
                        ('orig_url.txt', 'https://example.test/original.tar'), ('ntok.txt', '123'), ('mdp.txt', '12')])
        # Reverse TAR member order to prove numeric placeholder binding, including 10.
        for index in reversed(range(count)):
            suffix = 'png' if index % 2 == 0 else 'webp'
            members.append((f'{key}.{index}.{suffix}', image_bytes()))
    path = write_tar(root / 'shards/000/000000.tar', members)
    return {'collection': 'dcvlm', 'adapter': 'dcvlm_balanced_wds', 'root': str(root)}, path


def genund(root, members=None):
    path = write_tar(root / 'BLIP3o-Pretrain-Long-Caption/00000.tar', members or
                     [('a.png', image_bytes()), ('a.txt', 'A green square.')])
    return {'collection': 'genund', 'adapter': 'genund_caption_tar', 'root': str(root),
            'source': 'BLIP3o-Pretrain-Long-Caption'}, path


def records(root):
    manifest = build_shard_manifest(root)
    return list(iter_records(manifest, snapshot(root), {}, batch_size=1))


class UnifiedIngestTests(unittest.TestCase):
    def test_full_genund_root_routes_each_source_and_binds_mapping_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'source'
            directions={'BLIP3o-Pretrain-Long-Caption':'i2t','BLIP3o-Pretrain-Short-Caption':'t2i'}
            for source in directions:
                write_tar(dataset/source/'00000.tar',[('a.png',image_bytes()),('a.txt','A green square.')])
            spec={'collection':'genund','adapter':'genund_caption_tar','root':str(dataset),
                  'source_directions':directions,'preserve_all_captions':True}
            units=enumerate_inputs([spec])
            self.assertEqual(len(units),2)
            for unit in units:
                self.assertEqual(unit['source_directions'],directions)
                self.assertEqual(unit['caption_adapter_options'],
                                 {'preserve_all_captions':True,'direction':directions[unit['source']]})
            result=build([spec],root/'output',rules={'task_routing':'corpus.task_policy.v1'})
            self.assertEqual(result['scope'],'full_declared_inputs')
            actual=records(root/'output')
            self.assertEqual({row['source'] for row in actual},set(directions))
            for row in actual:
                direction=directions[row['source']]
                self.assertEqual(row['normalized']['category'],'single_'+direction)
                original=json.loads(row['raw']['original_supervision_json'])
                self.assertEqual(original['direction'],direction)
                self.assertEqual(original['image_loss_allowed'],direction=='t2i')
            changed=deepcopy(spec);changed['source_directions']['BLIP3o-Pretrain-Long-Caption']='t2i'
            self.assertNotEqual([unit['unit_id'] for unit in units],
                                [unit['unit_id'] for unit in enumerate_inputs([changed])])
            with self.assertRaisesRegex(ValueError,'plan changed'):
                build([changed],root/'output',rules={'task_routing':'corpus.task_policy.v1'})

    def test_genund_source_direction_map_requires_exact_coverage_and_no_global_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,_=genund(root/'source')
            short='BLIP3o-Pretrain-Short-Caption'
            write_tar(root/'source'/short/'00000.tar',[('a.png',image_bytes()),('a.txt','Short caption.')])
            spec.pop('source');long='BLIP3o-Pretrain-Long-Caption'
            for mapping in ({long:'i2t'},{long:'i2t',short:'t2i','not-mounted':'i2t'}):
                with self.subTest(mapping=mapping),self.assertRaisesRegex(ValueError,'exactly cover'):
                    enumerate_inputs([{**spec,'source_directions':mapping}])
            with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                enumerate_inputs([{**spec,'direction':'i2t','source_directions':{long:'i2t',short:'t2i'}}])
            with self.assertRaisesRegex(ValueError,'source-to-i2t/t2i mapping'):
                enumerate_inputs([{**spec,'source_directions':{long:'guess',short:'t2i'}}])

    def test_baseline_text_only_instruction_preserves_label_roles_and_text(self):
        # Real 005029.tar/001198.tar samples have this shape: instruction label,
        # six annotation members, no numbered image member, and no placeholder.
        # The contract applies to actual structure, not a hardcoded source name.
        conversation = ('human:A gas cylinder holds 34 L. 27(2/3) L is used.\n'
                        'How much is left?<EOCL>\ngpt:34 - 27(2/3) = 6(1/3).\n'
                        'Answer: 6(1/3) L.')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, source = balanced(root/'source', [
                (name, 'instruction', name, conversation, 0)
                for name in ('visualwebinstruct', 'allava', 'tqa')])
            before = file_sha(source)
            result = build([spec], root/'output', rows_per_file=2)
            self.assertEqual(result['statistics']['samples'], 3)
            self.assertEqual(result['statistics']['image_slots'], 0)
            for record in records(root/'output'):
                raw = record['raw']
                original = json.loads(raw['original_supervision_json'])
                self.assertEqual(raw['format'], 'text_conversation')
                self.assertEqual(raw['images'], [])
                self.assertEqual(record['normalized']['category'], 'pure_text')
                self.assertEqual(original['data_type'], 'instruction')
                self.assertEqual(original['source_annotations']['data_type.txt'], 'instruction')
                self.assertEqual(original['source_annotations']['conversations.txt'], conversation)
                self.assertEqual([part['role'] for part in raw['segments']], ['user', 'assistant'])
                self.assertEqual([part['text'] for part in raw['segments']],
                                 [part.split(':', 1)[1] for part in conversation.split('<EOCL>\n')])
            self.assertEqual(file_sha(source), before)

    def test_text_only_instruction_still_rejects_dangling_image_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root/'source', [
                ('a', 'instruction', 'visualwebinstruct', 'human:<image>Question?<EOCL>\ngpt:Answer', 0)])
            with self.assertRaisesRegex(ValueError, 'placeholder exceeds members'):
                build([spec], root/'output')
            self.assertFalse((root/'output/COMPLETE.json').exists())

    def test_balanced_original_roles_rounds_png_webp_and_interleaved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, source = balanced(root / 'source')
            before = file_sha(source)
            result = build([spec], root / 'output', rows_per_file=2)
            self.assertTrue(result['complete'])
            self.assertEqual(result['statistics']['samples'], 4)
            self.assertEqual(result['statistics']['image_slots'], 4)
            loaded = {record['raw']['source_key']: record for record in records(root / 'output')}
            self.assertEqual([loaded[key]['normalized']['category'] for key in 'abcd'],
                             ['single_i2t', 'multi_i2t', 'interleaved', 'pure_text'])
            for record in loaded.values():
                raw = record['raw']
                original = json.loads(raw['original_supervision_json'])
                self.assertEqual(raw['segments'], original['segments'])
                self.assertEqual(raw['text_hash'], text_fingerprint(raw['segments']))
                self.assertEqual(digest(load_original_record(record['locator'])), record['source_record_sha256'])
                for image in record['source_image_refs']:
                    self.assertEqual(read_asset_bytes(image), image_bytes())
                    self.assertEqual(image['offset'], image['offset_data'])
                    self.assertEqual(image['length'], image['size'])
            self.assertEqual(file_sha(source), before)

    def test_twelve_images_bind_numerically_and_never_flip_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source', [('many', 'mmdoc_mt', 'document',
                'human:' + '<image>'.join(str(i) for i in range(13)) + '<EOCL>\ngpt:Original document.', 12)])
            build([spec], root / 'output')
            record, = records(root / 'output')
            refs = record['source_image_refs']
            self.assertEqual([int(ref['member'].split('.')[1]) for ref in refs], list(range(12)))
            self.assertEqual(record['normalized']['category'], 'interleaved')
            self.assertEqual([x['image_index'] for x in record['raw']['segments'] if x['kind'] == 'image'], list(range(12)))

    def test_joint_collection_and_ai_caption_annotation_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, _ = balanced(root / 'balanced')
            second, _ = genund(root / 'genund')
            result = build([first, second], root / 'output')
            self.assertEqual(result['statistics']['samples'], 5)
            manifest = build_shard_manifest(root / 'output')
            self.assertEqual(manifest['collection'], 'mixed')
            rows = list(iter_records(manifest, snapshot(root / 'output'), {}, batch_size=1))
            record, = [row for row in rows if row['collection'] == 'genund']
            self.assertEqual(record['normalized']['category'], 'single_t2i')
            self.assertEqual(record['normalized']['provenance']['text']['origin'], 'ai')
            self.assertEqual(record['normalized']['provenance']['images'][0]['origin'], 'unknown')
            self.assertEqual(record['raw']['segments'][0]['kind'], 'text')
            self.assertEqual(record['prepared_adapter_status'], 'ready')
            self.assertEqual(len(record['caption_candidates']), 1)

    def test_genund_unsupported_annotation_is_quarantined_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = genund(root / 'source', [('a.png', image_bytes()), ('a.txt', 'Caption'),
                ('a.json', '{"question":"How many?","answer":"2"}')])
            build([spec], root / 'output')
            record, = records(root / 'output')
            self.assertEqual(record['pairing_state'], 'unresolved')
            self.assertEqual(record['prepared_adapter_status'], 'repair_metadata')
            self.assertIn('unsupported_task_annotation_requires_adapter', record['normalized']['issues'])

    def test_native_original_is_lossless_and_snapshot_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source')
            build([spec], root / 'first')
            native = {'collection': 'dcvlm', 'adapter': 'dcvlm_native', 'root': str(root / 'first')}
            build([native], root / 'second')
            first = {row['original_sample_id']: row for row in records(root / 'first')}
            for row in records(root / 'second'):
                previous = first[row['original_sample_id']]
                self.assertEqual(json.loads(row['raw']['original_supervision_json']), previous['raw'])
                self.assertEqual(row['raw']['segments'], previous['raw']['segments'])
                self.assertEqual(row['raw']['text_hash'], previous['raw']['text_hash'])

    def test_reimport_v2_requires_published_original_parquet_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source')
            build([spec], root / 'first')
            catalog = json.loads((root / 'first/catalog.json').read_text())
            path = root / 'first' / catalog[0]['path']
            value = bytearray(path.read_bytes()); value[20] ^= 1; path.write_bytes(value)
            native = {'collection': 'dcvlm', 'adapter': 'dcvlm_native', 'root': str(root / 'first')}
            with self.assertRaisesRegex(ValueError, 'published catalog'):
                build([native], root / 'second')
            self.assertFalse((root / 'second/COMPLETE.json').exists())

    def test_resume_verifies_sha_and_output_without_mutating_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, source = balanced(root / 'source')
            result = build([spec], root / 'output', rows_per_file=2)
            complete = root / 'output/COMPLETE.json'
            stamp = complete.stat().st_mtime_ns
            self.assertEqual(build([spec], root / 'output', rows_per_file=2), result)
            self.assertEqual(complete.stat().st_mtime_ns, stamp)
            state = source.stat()
            with source.open('r+b') as stream:
                stream.seek(513)
                old = stream.read(1)
                stream.seek(513)
                stream.write(bytes([old[0] ^ 1]))
            os.utime(source, ns=(state.st_atime_ns, state.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'source changed'):
                build([spec], root / 'output', rows_per_file=2)

    def test_partial_unit_resumes_and_limit_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source')
            unit, = enumerate_inputs([spec])
            partial = root / 'output/units' / unit['unit_id']
            partial.mkdir(parents=True)
            (partial / 'broken.partial').write_bytes(b'crash before unit receipt')
            result = build([spec], root / 'output', max_records=2)
            self.assertEqual(result['scope'], 'bounded_prefix')
            self.assertEqual(result['statistics']['samples'], 2)
            self.assertFalse((partial / 'broken.partial').exists())
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                build([spec], root / 'output')

    def test_byte_budget_flushes_and_committed_parquet_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source')
            with patch('corpus_pipeline.unified.ingest.MAX_BUFFER_BYTES', 1):
                build([spec], root / 'output')
                catalog = json.loads((root / 'output/catalog.json').read_text())
                self.assertEqual(sum(item['rows'] for item in catalog), 4)
                target = root / 'output' / catalog[0]['path']
                data = bytearray(target.read_bytes())
                data[20] ^= 1
                target.write_bytes(data)
                with self.assertRaisesRegex(ValueError, 'Parquet changed'):
                    build([spec], root / 'output')

    def test_full_genund_source_cannot_silently_skip_extracted_or_parquet_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, source = genund(root / 'source')
            (source.parent / 'another.jpg').write_bytes(image_bytes())
            with self.assertRaisesRegex(ValueError, 'full source enumeration cannot skip'):
                build([spec], root / 'output')
            spec['paths'] = [str(source.relative_to(root / 'source'))]
            result = build([spec], root / 'output')
            self.assertEqual(result['scope'], 'selected_inputs')

    def test_unknown_source_preview_and_duplicate_input_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, source = balanced(root / 'source')
            bad = deepcopy(spec)
            bad['paths'] = ['../outside.tar']
            with self.assertRaisesRegex(ValueError, 'traversal'):
                build([bad], root / 'output')
            with self.assertRaisesRegex(ValueError, 'duplicate input'):
                build([spec, spec], root / 'output')
            bad['paths'] = ['preview/preview.tar']
            write_tar(root / 'source/preview/preview.tar', [('a.txt', 'Preview')])
            with self.assertRaisesRegex(ValueError, 'never preview'):
                build([bad], root / 'output')
            unknown, _ = genund(root / 'genund')
            unknown['source'] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'unsupported gen-und source'):
                build([unknown], root / 'output')
            self.assertFalse((root / 'output/COMPLETE.json').exists())

    def test_bad_placeholder_or_duplicate_sample_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, _ = balanced(root / 'source', [('a', 'instruction', 'vqa', 'human:<image><image><EOCL>\ngpt:Answer', 1)])
            with self.assertRaisesRegex(ValueError, 'placeholder'):
                build([spec], root / 'output')
            self.assertFalse((root / 'output/COMPLETE.json').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example = ('a', 'text', 'source', 'human:Q<EOCL>\ngpt:A', 0)
            spec, _ = balanced(root / 'source', [example, ('b', 'text', 'source', 'human:Q<EOCL>\ngpt:A', 0), example])
            with self.assertRaisesRegex(ValueError, 'duplicate WebDataset key'):
                build([spec], root / 'output')


if __name__ == '__main__':
    unittest.main()
