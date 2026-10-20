"""Qwen3.5 chat-template construction with explicit per-span supervision.

Pretraining supervises every corpus token, including user turns; this is not
SFT and the prompt side carries as much language signal as the response. Only
the scaffolding fabricated here -- role headers and generated instructions --
is withheld from the loss, so the model never learns to emit our hand-written
wrappers as if they were data.

No thinking prefix is emitted. The base checkpoint has no <think> token, so the
prefix would tokenize into eight meaningless text fragments and teach a format
the model has no mechanism for.

The document BOS supplied by the packer is the first ``<|im_start|>``; the
opening span therefore starts at the role name. Instructions are selected by
hashing the immutable sample id, so a restart reproduces the exact same text.
"""
import hashlib

# Varied phrasings prevent the model from binding image synthesis to one
# literal string. They are deliberately plain: the corpus caption carries the
# content, the instruction only marks the direction of the turn.
T2I_INSTRUCTIONS = (
    'Generate an image from the following description: ',
    'Draw a picture that matches this description: ',
    'Create an image based on this caption: ',
    'Please render the following scene as an image: ',
    'Produce a photograph matching this text: ',
    'Make an image of the following: ',
    'Visualize this description as a picture: ',
    'Turn the following caption into an image: ',
    'Illustrate the following: ',
    'Synthesize an image for this prompt: ',
    'Show me an image of the following scene: ',
    'Using this description, generate the corresponding picture: ',
    'Render an image that depicts the following: ',
    'Based on the caption below, create a matching image: ',
    'Compose a picture from this description: ',
    'Generate the image described here: ',
)

I2T_INSTRUCTIONS = (
    'Describe this image.',
    'What is shown in this image?',
    'Write a caption for this image.',
    'Please describe what you see in the picture.',
    'Provide a description of this image.',
    'Explain the contents of this image.',
    'Caption the image above.',
    'Tell me what this picture shows.',
    'Give a short description of this image.',
    'What does this image depict?',
    'Summarize the visual content of this image.',
    'Describe the scene in the picture.',
    'Put the content of this image into words.',
    'Write a sentence describing this image.',
    'What can be seen in this photograph?',
    'Describe the image in detail.',
)

ROLE_USER = 'user\n'
ROLE_ASSISTANT = 'assistant\n'
TURN_END = '<|im_end|>\n'
IM_START = '<|im_start|>'


def choose(options, sample_id, salt):
    """Deterministic selection; a resumed run must rebuild identical text."""
    digest = hashlib.sha256(f'{salt}:{sample_id}'.encode()).digest()
    return options[int.from_bytes(digest[:8], 'big') % len(options)]


def text(value, supervised, synthetic):
    return dict(kind='text', text=value, supervised=supervised, synthetic=synthetic)


def corpus(value):
    """Real corpus text: always supervised, always counted."""
    return text(value, True, False)


def scaffold(value):
    """Fabricated wrapper: never supervised, never counted."""
    return text(value, False, True)


def closing():
    """Ends a message. Supervised so the model learns to stop, never counted."""
    return text(TURN_END, True, True)


def image(index):
    return dict(kind='image', image_index=index)


def header(role, first):
    """The packer's document BOS stands in for the first ``<|im_start|>``."""
    return scaffold(role if first else IM_START + role)


def raw_direction(segments):
    """Caption direction from the stored order, before any templating."""
    kinds = [s['kind'] for s in segments]
    if kinds == ['text', 'image']:
        return 'T2I'
    if kinds == ['image', 'text']:
        return 'I2T'
    return None


def caption_spans(segments, direction, sample_id):
    body = next(s for s in segments if s['kind'] == 'text')
    picture = next(s for s in segments if s['kind'] == 'image')
    spans = [header(ROLE_USER, True)]
    if direction == 'T2I':
        spans.append(scaffold(choose(T2I_INSTRUCTIONS, sample_id, 't2i')))
        spans.append(corpus(body['text']))
        spans.append(closing())
        spans.append(header(ROLE_ASSISTANT, False))
        spans.append(image(picture['image_index']))
    else:
        spans.append(image(picture['image_index']))
        spans.append(scaffold(choose(I2T_INSTRUCTIONS, sample_id, 'i2t')))
        spans.append(closing())
        spans.append(header(ROLE_ASSISTANT, False))
        spans.append(corpus(body['text']))
    spans.append(closing())
    return spans


def conversation_spans(segments):
    """Group stored segments by turn and wrap each in its own message.

    Without this the stored turns are concatenated with no separator at all,
    which teaches an incorrect document structure.
    """
    spans, current, first = [], None, True
    for segment in segments:
        role = segment.get('role') or 'user'
        turn = segment.get('turn')
        if current != (role, turn):
            if current is not None:
                spans.append(closing())
            spans.append(header(ROLE_ASSISTANT if role == 'assistant' else ROLE_USER, first))
            current, first = (role, turn), False
        if segment['kind'] == 'image':
            spans.append(image(segment['image_index']))
        elif segment['text']:
            spans.append(corpus(segment['text']))
    if current is not None:
        spans.append(closing())
    return spans


def natural_spans(segments):
    """Interleaved documents stay natural documents; no chat wrapper."""
    return [image(s['image_index']) if s['kind'] == 'image' else corpus(s['text'])
            for s in segments if s['kind'] == 'image' or s['text']]


def build(row):
    """Return templated spans for one corpus row.

    Captions and conversations become chat messages; interleaved documents keep
    their natural form. Spans are plain dicts so the caller can batch-tokenize
    every text at once and keep the per-span supervision flags.

    The third result says whether the packer should open this document with
    ``<|im_start|>``. Natural documents are plain text and must not be dressed
    as a conversation, so they receive no opener at all.
    """
    segments = row['segments']
    fmt = row.get('format')
    if fmt == 'caption':
        direction = raw_direction(segments)
        if direction is not None:
            return caption_spans(segments, direction, row['sample_id']), direction, True
        return natural_spans(segments), None, False
    if fmt in ('conversation', 'text_conversation'):
        return conversation_spans(segments), None, True
    return natural_spans(segments), None, False
