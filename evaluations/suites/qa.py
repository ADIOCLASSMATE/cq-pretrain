"""Greedy single-image answers for GQA, MME, POPE and TextVQA (qa) or VQAv2 (vqav2)."""
from collections import defaultdict
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import string
import sys
import types
from zipfile import ZipFile

import torch.distributed as dist

from utils.checkpoint_catalog import durable_json

CHUNK = 32
EXPECTED = {'qa': {'gqa': 12578, 'mme': 2374, 'pope': 9000, 'textvqa': 5000}, 'vqav2': {'vqav2': 214354}}
LMMS_EVAL_REVISION = 'c969af54173a5b5393a6dfe8304dd87423fc9388'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run(runtime, out, suite, benchmarks, limit=0):
    """Each rank answers whole chunks; any finished chunk survives preemption."""
    if (out / 'RESULTS.json').is_file():
        return
    data = Path(benchmarks) / suite
    ready = json.loads((data / 'READY.json').read_text())
    if hashlib.sha256((data / 'questions.jsonl').read_bytes()).hexdigest() != ready['questions_sha256']:
        raise ValueError(f'{suite} questions changed')
    questions = rows(data / 'questions.jsonl')
    for row in questions:
        row['image'] = str(data / row['image'])
    if limit:
        questions = questions[:limit]
    rank, world = dist.get_rank(), dist.get_world_size()
    chunks = (len(questions) + CHUNK - 1) // CHUNK
    folder = out / 'raw'
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(rank, chunks, world):
        path = folder / f'{k:05d}.json'
        if path.exists():
            continue
        answers = []
        for case in questions[k * CHUNK:(k + 1) * CHUNK]:
            result = runtime.answer(case)
            answers.append(dict(id=case['id'], text=result['text'], stop_reason=result['stop_reason'],
                                token_ids=result['token_ids']))
        durable_json(path, answers)
        if k % 128 == 0:
            print(f'step {runtime.step} {suite} chunk {k}/{chunks}', flush=True)
    dist.barrier()
    if rank == 0:
        predictions = [p for k in range(chunks) for p in json.loads((folder / f'{k:05d}.json').read_text())]
        if [p['id'] for p in predictions] != [q['id'] for q in questions]:
            raise ValueError('Predictions do not cover the questions in order')
        if limit:
            durable_json(out / 'SMOKE.json', dict(task=suite, **runtime.describe(), answers=predictions))
            print(f'step {runtime.step} {suite} smoke complete', flush=True)
            dist.barrier()
            return
        fields = vqav2_report(out, data, ready, predictions) if suite == 'vqav2' else qa_report(out, data, ready, predictions)
        if fields['cases'] != EXPECTED[suite]:
            raise ValueError(f'Unexpected {suite} case counts: {fields["cases"]}')
        durable_json(out / 'RESULTS.json', dict(task=suite, **runtime.describe(), **fields))
        print(f'step {runtime.step} {suite} complete', flush=True)
    dist.barrier()


def _lmms_eval(root):
    # Import the benchmark plugins directly; the lmms-eval CLI registry would also import model backends.
    sys.path.insert(0, str(root / 'scorer-deps'))
    kit = root / 'lmms-eval'
    for package, path in [('lmms_eval', kit / 'lmms_eval'), ('lmms_eval.tasks', kit / 'lmms_eval/tasks')]:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    return [importlib.import_module(f'lmms_eval.tasks.{name}.utils') for name in ('mme', 'pope', 'textvqa')]


def score(root, answers, references):
    mme, pope, textvqa = _lmms_eval(root)
    gqa, ocr, by_mme, by_pope = [], [], defaultdict(list), defaultdict(list)
    normalize = lambda s: s.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    for pred, ref in zip(answers, references, strict=True):
        if pred['id'] != ref['id']:
            raise ValueError('Prediction and reference order differ')
        text = pred['text']
        if ref['dataset'] == 'gqa':
            gqa.append(float(normalize(text) == normalize(ref['answer'])))
        elif ref['dataset'] == 'mme':
            for k, v in mme.mme_process_results(ref, [text]).items():
                by_mme[k].append(v)
        elif ref['dataset'] == 'pope':
            by_pope[ref['category']].append(pope.pope_process_results(ref, [text])['pope_accuracy'])
        elif ref['dataset'] == 'textvqa':
            ocr.append(textvqa.textvqa_process_results(dict(ref, answers=list(ref['answers'])), [text])['exact_match'])
        else:
            raise ValueError(ref['dataset'])
    metrics = dict(gqa_accuracy=statistics.mean(gqa), textvqa_vqa_accuracy=statistics.mean(ocr),
                   **{k: mme.mme_aggregate_results(v) for k, v in by_mme.items()})
    metrics['mme_total'] = metrics['mme_perception_score'] + metrics['mme_cognition_score']

    def pope_scores(values):
        return dict(accuracy=pope.pope_aggregate_accuracy(values), precision=pope.pope_aggregate_precision(values),
                    recall=pope.pope_aggregate_recall(values), f1=pope.pope_aggregate_f1_score(values),
                    yes_ratio=pope.pope_aggregate_yes_ratio(values))
    metrics['pope_by_split'] = {name: pope_scores(v) for name, v in by_pope.items()}
    metrics['pope_macro_f1'] = statistics.mean(x['f1'] for x in metrics['pope_by_split'].values())
    metrics['pope_all'] = pope_scores([x for v in by_pope.values() for x in v])
    return metrics


def qa_report(out, data, ready, predictions):
    if hashlib.sha256((data / 'references.jsonl').read_bytes()).hexdigest() != ready['references_sha256']:
        raise ValueError('QA references changed')
    counts = defaultdict(int)
    for p in predictions:
        counts[p['id'].split('/', 1)[0]] += 1
    metrics = score(data, predictions, rows(data / 'references.jsonl'))
    path = out / 'predictions.jsonl'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in predictions))
    temporary.replace(path)
    return dict(cases=dict(counts), metrics=metrics, questions_sha256=ready['questions_sha256'],
                references_sha256=ready['references_sha256'], predictions=str(path), scorer_revision=LMMS_EVAL_REVISION)


class _Records:
    def __init__(self, values):
        self.qa = {row['question_id']: row for row in values}

    def getQuesIds(self):
        return list(self.qa)


def vqav2_report(out, data, ready, predictions):
    """The official VQA scorer (normalization and 10-annotator min(#humans/3, 1))."""
    raw = [dict(question_id=int(p['id'].split('/', 1)[1]), answer=p['text']) for p in predictions]
    archive = data / 'v2_Annotations_Val_mscoco.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ready['annotation_archive_sha256']:
        raise ValueError('VQAv2 annotation archive changed')
    if hashlib.sha256((data / 'vqaEval.py').read_bytes()).hexdigest() != ready['python3_scorer_sha256']:
        raise ValueError('VQAv2 scorer changed')
    with ZipFile(archive) as z, z.open('v2_mscoco_val2014_annotations.json') as f:
        annotations = json.load(f)['annotations']
    sys.path.insert(0, str(data))
    from vqaEval import VQAEval
    gt, results = _Records(annotations), _Records(raw)
    if gt.qa.keys() != results.qa.keys():
        raise ValueError('VQAv2 predictions do not match the annotations')
    scorer = VQAEval(gt, results, n=4)
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        scorer.evaluate()
    path = out / 'predictions.json'
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(raw, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(path)
    return dict(cases={'vqav2': len(raw)},
                metrics={'vqav2_accuracy_pct': scorer.accuracy['overall'],
                         'per_answer_type_pct': scorer.accuracy['perAnswerType'],
                         'per_question_type_pct': scorer.accuracy['perQuestionType']},
                predictions=str(path), scorer_revision=ready['official_scorer_revision'],
                annotation_archive_sha256=ready['annotation_archive_sha256'], questions_sha256=ready['questions_sha256'])
