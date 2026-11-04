"""Contextual task-quality review using the existing resumable single-writer farm.

This creates a separate run. It never mutates visual-screening or human labels.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from corpus_pipeline.codex_review.farm import (
    Farm, atomic_json, canonical, digest, exclusive, file_sha, readonly_status,
)


class ContextFarm(Farm):
    def load_worker(self):
        if self.worker is None:
            self.worker = importlib.import_module('corpus_pipeline.context_review.worker')
        return self.worker

    def claim(self, batch_size, *, probe=False):
        # Keep whole long documents, but do not combine several into one request.
        rows = self.db.execute("""SELECT input_json FROM items WHERE status IN ('pending','retry')
            AND retry_not_before<=? ORDER BY attempts DESC,queue_order,item_id LIMIT ?""",
                               (self.clock(), batch_size)).fetchall()
        chars = 0
        for row in rows:
            context = json.loads(row[0])['source_context']
            chars += sum(len(value) for key in ('caption_texts', 'conversation_texts') for value in context[key])
            chars += sum(len(segment['text']) for segment in context['interleaved_segments'])
        return super().claim(1 if chars > 80000 else batch_size, probe=probe)

    def pin_runtime(self):
        worker = self.load_worker()
        packages = ('corpus_pipeline.codex_review.farm', 'corpus_pipeline.codex_review.inputs',
                    'corpus_pipeline.codex_review.worker', 'corpus_pipeline.context_review.worker',
                    'corpus_pipeline.context_review.extract', 'corpus_pipeline.context_review.run')
        files = {}
        for name in packages:
            path = Path(importlib.import_module(name).__file__).resolve()
            files[str(path)] = file_sha(path)
        binding = {'code': files, 'config': self.config, 'model': worker.MODEL,
                   'reasoning_effort': worker.EFFORT,
                   'prompt_version': worker.PROMPT_VERSION, 'schema_sha256': digest(worker.SCHEMA)}
        previous = self.metadata('runtime_binding')
        if previous is not None and previous != binding:
            raise ValueError('context runtime changed; explicit new run required')
        with self.db:
            self.put('runtime_binding', binding)
        # Keep exact source snapshots for the immutable audit, even after later development.
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

    def _audit_context(self):
        worker = self.load_worker()
        checked = 0
        for batch in self.db.execute("""SELECT * FROM batches WHERE batch_id IN
              (SELECT DISTINCT batch_id FROM items WHERE status='succeeded')"""):
            directory = Path(batch['directory'])
            items = json.loads((directory / 'input.json').read_text())
            metadata = json.loads(batch['metadata_json'])
            attached_ids = metadata['attached_item_ids']
            attached = [item for item in items if item['item_id'] in attached_ids]
            if (directory / 'prompt.txt').read_text() != worker.build_prompt(attached):
                raise ValueError('prompt context differs from canonical input')
            if json.loads((directory / 'schema.json').read_text()) != worker.SCHEMA:
                raise ValueError('context output schema differs')
            mapping = {item['item_id']: item for item in items}
            for row in self.db.execute("SELECT item_id,result_json FROM items WHERE batch_id=? AND status='succeeded'",
                                       (batch['batch_id'],)):
                worker.validate_result(json.loads(row['result_json']), row['item_id'], mapping[row['item_id']])
                checked += 1
        successes = self.db.execute("SELECT count(*) FROM items WHERE status='succeeded'").fetchone()[0]
        if checked != successes:
            raise ValueError('context audit coverage mismatch')
        return checked

    def audit(self, *, require_complete=True, export_manifest=None):
        try:
            checked = self._audit_context()
            result = super().audit(require_complete=require_complete, export_manifest=export_manifest)
        except Exception as error:
            atomic_json(self.root / 'AUDIT.json', {
                'mode': 'strict', 'passed': False, 'complete_coverage': False,
                'error': type(error).__name__ + ': ' + str(error), 'skipped_gates': [],
                'time': self.clock(), 'scope': 'contextual training review'})
            raise
        result.update(context_items_checked=checked, context_prompt_binding_checked=True,
                      original_screening_modified=False, human_decisions_modified=False)
        atomic_json(self.root / 'AUDIT.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'status', 'audit', 'export'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--initial-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.action == 'status':
        print(canonical(readonly_status(args.root)))
        return
    with exclusive(args.root):
        farm = ContextFarm(args.root)
        try:
            if args.action == 'init':
                if args.manifest is None:
                    parser.error('--manifest is required for init')
                result = farm.initialize(args.manifest)
            elif args.action == 'run':
                if args.batch_size > 4 or args.workers > 32:
                    parser.error('context contract requires batch_size<=4 and workers<=32')
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
