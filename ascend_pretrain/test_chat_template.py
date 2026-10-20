"""Chat-template construction, supervision masking and token accounting."""
import numpy as np

from .input_pipeline import chat


def ids(spans):
    return [(s['kind'], s.get('text'), s.get('supervised'), s.get('synthetic')) for s in spans]


def texts(spans):
    return ''.join(s['text'] for s in spans if s['kind'] == 'text')


def caption_row(direction, body='a red bicycle', sample_id='abc'):
    picture = dict(kind='image', text=None, image_index=0, role=None, turn=None)
    text = dict(kind='text', text=body, image_index=None, role=None, turn=None)
    segments = [text, picture] if direction == 'T2I' else [picture, text]
    return dict(sample_id=sample_id, format='caption', segments=segments)


def test_t2i_puts_caption_in_user_turn_and_image_in_assistant():
    spans, direction, templated = chat.build(caption_row('T2I'))
    assert templated
    assert direction == 'T2I'
    assert spans[-1]['kind'] == 'text' and spans[-1]['text'] == chat.TURN_END
    assert spans[-2]['kind'] == 'image'
    rendered = texts(spans)
    assert rendered.startswith('user\n')
    assert 'a red bicycle' in rendered
    # No thinking prefix: the base checkpoint has no <think> token.
    assert '<think>' not in rendered
    assert rendered.index('a red bicycle') < rendered.index(chat.ROLE_ASSISTANT)


def test_i2t_puts_image_in_user_turn_and_caption_in_assistant():
    spans, direction, templated = chat.build(caption_row('I2T'))
    assert templated
    assert direction == 'I2T'
    assert spans[1]['kind'] == 'image'
    rendered = texts(spans)
    assert '<think>' not in rendered
    assert rendered.index(chat.ROLE_ASSISTANT) < rendered.index('a red bicycle')


def test_only_corpus_text_is_supervised_and_counted():
    spans, _, _ = chat.build(caption_row('T2I'))
    supervised = [s['text'] for s in spans if s['kind'] == 'text' and s['supervised']]
    counted = [s['text'] for s in spans if s['kind'] == 'text' and not s['synthetic']]
    # The caption and every message terminator are learned.
    assert 'a red bicycle' in supervised
    assert chat.TURN_END in supervised
    # Role headers, instructions and the thinking prefix are never learned.
    assert not any(t.startswith('user') or t.startswith(chat.IM_START) for t in supervised)
    # Only real corpus text may perturb the published token audit.
    assert counted == ['a red bicycle']


def test_instruction_is_deterministic_and_varied():
    first = chat.build(caption_row('T2I', sample_id='sample-one'))[0]
    again = chat.build(caption_row('T2I', sample_id='sample-one'))[0]
    assert texts(first) == texts(again), 'resume must rebuild identical text'
    seen = {texts(chat.build(caption_row('T2I', sample_id=f'id-{i}'))[0]).split('user\n')[1].split('a red')[0]
            for i in range(400)}
    assert len(seen) == len(chat.T2I_INSTRUCTIONS)


def test_conversation_turns_are_separated():
    row = dict(sample_id='c', format='conversation', segments=[
        dict(kind='image', text=None, image_index=0, role='user', turn=0),
        dict(kind='text', text='What is this?', image_index=None, role='user', turn=0),
        dict(kind='text', text='A cat.', image_index=None, role='assistant', turn=1)])
    spans, _, templated = chat.build(row)
    assert templated
    rendered = texts(spans)
    # Without the template these two turns were concatenated with no separator.
    assert 'What is this?' + chat.TURN_END in rendered
    assert rendered.index('What is this?') < rendered.index(chat.IM_START + chat.ROLE_ASSISTANT)
    assert rendered.index(chat.ROLE_ASSISTANT) < rendered.index('A cat.')
    supervised = [s['text'] for s in spans if s['kind'] == 'text' and s['supervised']]
    assert 'What is this?' in supervised, 'pretraining supervises the user turn too'
    assert 'A cat.' in supervised


def test_multi_turn_alternation_opens_each_message_once():
    segments = []
    for turn in range(4):
        role = 'user' if turn % 2 == 0 else 'assistant'
        segments.append(dict(kind='text', text=f'msg{turn}', image_index=None, role=role, turn=turn))
    spans, _, _ = chat.build(dict(sample_id='m', format='text_conversation', segments=segments))
    rendered = texts(spans)
    assert rendered.count(chat.TURN_END) == 4
    # Three explicit openers; the packer's document BOS supplies the first.
    assert rendered.count(chat.IM_START) == 3
    assert '<think>' not in rendered


def test_interleaved_documents_keep_natural_form():
    segments = [dict(kind='text', text='A', image_index=None, role=None, turn=None),
                dict(kind='image', text=None, image_index=0, role=None, turn=None),
                dict(kind='text', text='B', image_index=None, role=None, turn=None)]
    spans, direction, templated = chat.build(dict(sample_id='i', format='interleaved', segments=segments))
    assert direction is None
    assert not templated, 'natural documents must not be opened as a conversation'
    assert texts(spans) == 'AB', 'natural documents receive no chat scaffolding'
    assert all(s['supervised'] for s in spans if s['kind'] == 'text')
    assert chat.IM_START not in texts(spans)


def test_empty_corpus_spans_are_dropped():
    segments = [dict(kind='text', text='', image_index=None, role=None, turn=None),
                dict(kind='image', text=None, image_index=0, role=None, turn=None)]
    spans, _, _ = chat.build(dict(sample_id='e', format='interleaved', segments=segments))
    assert [s['kind'] for s in spans] == ['image']


import unittest

import numpy as np

from .config import tiny_config
from .input_pipeline.packing import Packer


class Documents:
    """Minimal document source exposing the packer's stream contract."""
    images_preprocessed = False

    def __init__(self, docs):
        self.docs, self.index = docs, 0

    def peek(self):
        return self.docs[self.index] if self.index < len(self.docs) else None

    def advance(self):
        self.index += 1

    def prefetch_images(self, *args):
        pass

    def state_dict(self):
        return {}

    def close(self):
        pass


TOKENS = dict(bos=1, eos=2, boi=3, eoi=4, image=5, text_mask=6)


def document(ident, ids, templated, supervised=True):
    return dict(id=ident, format='x', direction=None, templated=templated,
                elements=[dict(type='text', input_ids=np.asarray(ids, np.int32),
                               supervised=supervised)])


def pack(docs, length=32):
    cfg = tiny_config(vision_encoder='none', renderer='none', preserve_documents=False)
    batch, receipt = Packer(Documents(docs), cfg, length, TOKENS).next()
    return batch, receipt


class OpenerContract(unittest.TestCase):
    def test_templated_document_opens_with_im_start(self):
        batch, _ = pack([document('a', [10, 11, 12], templated=True)])
        self.assertEqual(batch.ids[0, 0].item(), TOKENS['bos'])
        self.assertEqual(batch.ids[0, 1:4].tolist(), [10, 11, 12])

    def test_natural_document_has_no_opener(self):
        batch, _ = pack([document('a', [10, 11, 12], templated=False)])
        # The interleaved document must begin at its own first token.
        self.assertEqual(batch.ids[0, :3].tolist(), [10, 11, 12])
        self.assertEqual(batch.ids[0, 3].item(), TOKENS['eos'])

    def test_mixed_pack_opens_only_the_templated_document(self):
        batch, receipt = pack([document('a', [10, 11], templated=False),
                               document('b', [20, 21], templated=True)])
        self.assertTrue(all(d['completed'] for d in receipt['documents']))
        # natural: 10 11 EOS, then templated: BOS 20 21 EOS
        self.assertEqual(batch.ids[0, :7].tolist(),
                         [10, 11, TOKENS['eos'], TOKENS['bos'], 20, 21, TOKENS['eos']])

    def test_unsupervised_spans_are_excluded_from_the_loss_index(self):
        batch, _ = pack([document('a', [10, 11, 12], templated=False, supervised=False)])
        body = set(range(3))
        self.assertFalse(body & set(batch.text_indices.tolist()),
                         'scaffolding must never be supervised')
        # The document terminator is still learned.
        self.assertIn(3, batch.text_indices.tolist())


class QueryPruning(unittest.TestCase):
    """Scaffolding is fed as content but must never get its own readout."""

    def plan_for(self, supervised):
        import torch
        from .config import tiny_config
        from .data import fake_batch
        from .mainline import build_plan
        cfg = tiny_config()
        batch = fake_batch(cfg, 64, 1, 1)
        if supervised is not None:
            batch.text_indices = supervised
        return build_plan(batch, cfg), batch, cfg

    def test_queries_cover_exactly_the_supervised_positions(self):
        import torch
        plan, batch, cfg = self.plan_for(None)
        full = len(plan.query_positions)
        every = batch.text_indices
        # Withhold a third of the positions, as chat scaffolding does.
        keep = every[torch.arange(len(every)) % 3 != 0]
        plan2, batch2, _ = self.plan_for(keep)
        self.assertLess(len(plan2.query_positions), full,
                        'unsupervised positions must not receive a query')
        image = batch2.image_queries[0]
        text_q = plan2.query_positions[~image[plan2.query_positions]]
        self.assertEqual(set(text_q.tolist()), set(keep.tolist()),
                         'text queries must equal the supervised positions')

    def test_empty_supervision_keeps_a_live_graph(self):
        import torch
        plan, _, _ = self.plan_for(torch.empty(0, dtype=torch.long))
        self.assertTrue(len(plan.query_positions) > 0,
                        'a fully unsupervised pack still needs a DDP-safe graph')
