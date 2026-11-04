"""Independent broad-retention recheck of exactly 208 first-pass exclusions."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

from corpus_pipeline.codex_review.farm import atomic_json, canonical, digest, exclusive, file_sha, readonly_status
from corpus_pipeline.context_review.run import ContextFarm

EXPECTED = 208


class LenientFarm(ContextFarm):
    def load_worker(self):
        if self.worker is None:
            self.worker = importlib.import_module('corpus_pipeline.context_review.lenient_worker')
        return self.worker

    def pin_runtime(self):
        worker = self.load_worker()
        packages = ('corpus_pipeline.codex_review.farm', 'corpus_pipeline.codex_review.inputs',
                    'corpus_pipeline.codex_review.worker', 'corpus_pipeline.context_review.worker',
                    'corpus_pipeline.context_review.extract', 'corpus_pipeline.context_review.run',
                    'corpus_pipeline.context_review.lenient_worker', 'corpus_pipeline.context_review.lenient_run')
        files = {str(Path(importlib.import_module(name).__file__).resolve()):
                 file_sha(importlib.import_module(name).__file__) for name in packages}
        binding = {'code': files, 'config': self.config, 'model': worker.MODEL,
                   'reasoning_effort': worker.EFFORT, 'prompt_version': worker.PROMPT_VERSION,
                   'schema_sha256': digest(worker.SCHEMA)}
        previous = self.metadata('runtime_binding')
        if previous is not None and previous != binding:
            raise ValueError('lenient runtime changed; explicit new run required')
        with self.db:
            self.put('runtime_binding', binding)
        for path, sha in files.items():
            snapshot = self.root / 'code_snapshots' / (sha + '.py')
            snapshot.parent.mkdir(exist_ok=True)
            if snapshot.exists():
                if file_sha(snapshot) != sha:
                    raise ValueError('code snapshot hash mismatch')
            else:
                with snapshot.open('xb') as destination:
                    destination.write(Path(path).read_bytes())
        return worker

    def initialize(self, manifest):
        selection = json.loads((self.root / 'SELECTION.json').read_text())
        if (selection.get('passed') is not True or selection.get('records') != EXPECTED
                or file_sha(manifest) != selection['output']['sha256']):
            raise ValueError('lenient input selection invalid')
        result = super().initialize(manifest)
        if result['total'] != EXPECTED:
            raise ValueError('expected exactly 208 excluded source items')
        return result

    def audit(self, *, require_complete=True, export_manifest=None):
        source = self.metadata('source')
        selection = json.loads((self.root / 'SELECTION.json').read_text())
        if (source.get('count') != EXPECTED or selection.get('passed') is not True
                or source['sha256'] != selection['output']['sha256']):
            raise ValueError('lenient selection/source binding mismatch')
        result = super().audit(require_complete=require_complete, export_manifest=export_manifest)
        result.update(initial_exclusion_scope_checked=True, selection_sha256=file_sha(self.root / 'SELECTION.json'),
                      original_context_results_modified=False)
        atomic_json(self.root / 'AUDIT.json', result)
        return result


def prepare(parent, root):
    from corpus_pipeline.context_review.report import load_completed
    parent, root = Path(parent).resolve(strict=True), Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite existing recheck run')
    rows, bindings = load_completed(parent)
    selected = [row for row in rows if row['result']['recommendation'] in {'exclude_pair', 'drop_image_keep_text'}]
    if len(selected) != EXPECTED:
        raise ValueError('expected exactly 208 initial exclusions')
    # Preserve every original candidate field. Only the subset/order changes;
    # no initial model verdict is passed into the source context or prompt.
    selected.sort(key=lambda row: (row['candidate']['queue_order'], row['candidate']['item_id']))
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'candidates.jsonl'
    with target.open('xb') as stream:
        for row in selected:
            stream.write((canonical(row['candidate']) + '\n').encode())
    selection = {'schema_version': 'corpus.context_review.exclusion_recheck.selection.v1', 'passed': True,
                 'records': EXPECTED, 'initial_source_bindings': bindings,
                 'selection': 'initial recommendation in exclude_pair/drop_image_keep_text',
                 'source_inputs_unchanged': True, 'initial_results_in_prompt': False,
                 'initial_recommendations': dict(Counter(row['result']['recommendation'] for row in selected)),
                 'task_types': dict(Counter(row['candidate']['task_type'] for row in selected)),
                 'first_eight_review_numbers': [row['candidate']['review_number'] for row in selected[:8]],
                 'records_selected': [{'item_id': row['candidate']['item_id'], 'review_number': row['candidate']['review_number'],
                                       'candidate_sha256': digest(row['candidate']), 'initial_result_sha256': digest(row['result']),
                                       'initial_recommendation': row['result']['recommendation']} for row in selected],
                 'output': {'path': str(target), 'sha256': file_sha(target), 'bytes': target.stat().st_size}}
    atomic_json(root / 'SELECTION.json', selection)
    config = json.loads((parent / 'config.json').read_text())
    config.update(timeout_seconds=300, increase_every=1)
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'CONTRACT.json', {'expected_items': EXPECTED, 'model': 'gpt-5.6-sol', 'reasoning_effort': 'low',
        'prompt_version': 'dcvlm.context_training_review.v2_lenient_exclusions', 'max_workers': 32,
        'max_images_per_batch': 4, 'audit_mode': 'strict', 'real_image_attachments_required': True,
        'source_inputs_unchanged': True, 'original_model_and_human_decisions_unchanged': True,
        'policy': 'Pass unless clearly wrong. Unverified names/details are not contradictions; visible useful broad scenes survive face blur; related interleaved illustrations may be low-information. Exclusion requires independently severe information destruction or concrete associated-text contradiction.',
        'source_selection_sha256': file_sha(root / 'SELECTION.json')})
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'init', 'run', 'status', 'audit', 'export'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--initial-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.parent is None:
            parser.error('--parent required for prepare')
        result = prepare(args.parent, args.root)
        print(canonical({key: result[key] for key in ('records', 'initial_recommendations', 'task_types', 'first_eight_review_numbers', 'output')}))
        return
    if args.action == 'status':
        print(canonical(readonly_status(args.root)))
        return
    with exclusive(args.root):
        farm = LenientFarm(args.root)
        try:
            if args.action == 'init':
                result = farm.initialize(args.root / 'candidates.jsonl')
            elif args.action == 'run':
                if args.batch_size > 4 or args.workers > 32:
                    parser.error('require batch_size<=4, workers<=32')
                result = farm.run(workers=args.workers, initial_workers=args.initial_workers,
                                  batch_size=args.batch_size, max_batches=args.max_batches)
            elif args.action == 'audit':
                result = farm.audit(require_complete=not args.allow_incomplete)
            else:
                result = farm.export(shard_size=256, require_complete=not args.allow_incomplete)
            print(canonical(result))
        finally:
            farm.close()


if __name__ == '__main__':
    main()
