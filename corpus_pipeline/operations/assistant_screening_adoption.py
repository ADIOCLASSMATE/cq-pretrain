"""Explicit, hash-bound adoption of directly observed assistant visual reviews.

This is a new evidence-authority extension, not a Qwen result or a modification
of the frozen execution. Original API files and ledger remain unchanged. Apply
uses the frozen history assembler and receipt sealer under both existing locks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import struct
import sys

AUTHORITY = 'openai_assistant_visual_review'
SCHEMA = 'corpus.operational.screening_adoption.v2'
USER_INSTRUCTION = '你直接帮我把这几条审查算了'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def bound(path, expected=None):
    path = Path(path).resolve(strict=True)
    value = {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}
    require(expected is None or value['sha256'] == expected, 'Source SHA mismatch: ' + str(path))
    return value


def immutable_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Immutable adoption evidence changed: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def snapshot_file(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    before = bound(source)
    if not target.exists():
        temporary = target.with_name(target.name + '.tmp')
        shutil.copyfile(source, temporary)
        require(sha(temporary) == before['sha256'], 'Snapshot copy mismatch')
        os.replace(temporary, target)
    bound(target, before['sha256'])
    require(bound(source) == before, 'Source changed during snapshot')
    return bound(target)


@contextmanager
def existing_writer_locks(run):
    """Identical lock APIs/order to coordinator and same-host API writer."""
    owner = read(run / 'api_review/OWNER.json')
    require(owner.get('host') == socket.gethostname(), 'API ledger belongs to another host')
    with (run / '.coordinator.lock').open('a') as coordinator:
        fcntl.lockf(coordinator, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (run / 'api_review/.lock').open('a') as api:
                fcntl.flock(api, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
        finally:
            fcntl.lockf(coordinator, fcntl.LOCK_UN)


def connect_ledger(run):
    con = sqlite3.connect('file:' + str(run / 'api_review/index.sqlite') + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    return con


def request_snapshot(run):
    with connect_ledger(run) as con:
        rows = [dict(row) for row in con.execute('''
          SELECT a.asset_id,a.input_sha,a.state AS asset_state,
                 r.state AS request_state,r.request_sha,r.receipt_path,r.receipt_sha
          FROM assets a JOIN requests r USING(asset_id)
          WHERE r.role='primary' ORDER BY a.asset_id''')]
        total = con.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    require(len(rows) == total and len({r['asset_id'] for r in rows}) == total,
            'Missing or duplicate primary requests')
    return rows


def normalized_qwen(row, request):
    """Run the original assembler's checks and original normalization unchanged."""
    from corpus_pipeline.dedup import decisions as legacy
    require(request['request_state'] == 'ok' and request['receipt_path'] and
            legacy.ASSET.fullmatch(str(request['receipt_sha'])), 'Qwen request is not successful')
    require(row['state'] == request['asset_state'] == 'done' and
            row['input_sha256'] == request['input_sha'], 'Qwen ledger/input mismatch')
    raw = json.loads(row['decision_json'])
    primary = (raw.get('reviews') or {}).get('primary') or {}
    require(primary.get('assessment') in {'likely_ai_generated', 'likely_non_ai', 'possible_ai_edit', 'uncertain'}
            and primary.get('decision') in {'admit', 'reject', 'pending'}, 'Missing completed Qwen visual assessment')
    return legacy.normalize_qwen(row)


def normalize_assistant(review, evidence):
    """Same visual decision gates, without constructing a ModelClient receipt."""
    from corpus_pipeline.api_review.aggregate import validate_parsed
    from corpus_pipeline.production.image_screening import _cue_count
    from corpus_pipeline.dedup import decisions as legacy
    aid = review.get('asset_id'); parsed = review.get('parsed')
    require(legacy.ASSET.fullmatch(str(aid)), 'Invalid assistant asset ID')
    require(review.get('completed') is True, 'Image has not been directly reviewed')
    require(isinstance(review.get('viewed_at_utc'), str), 'Missing observation timestamp')
    stamp = datetime.fromisoformat(review['viewed_at_utc'].replace('Z', '+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0,
            'Observation timestamp must be UTC')
    errors = validate_parsed(parsed)
    require(not errors, 'Invalid assistant visual schema: ' + ','.join(errors))
    require(len(parsed['visible_evidence']) <= 3 and len(parsed['counterevidence']) <= 2
            and len(parsed['reason']) <= 120, 'Assistant response exceeds original v4 prompt bounds')
    require(parsed['assessment'] != 'likely_non_ai' or parsed['confidence'] == 'medium',
            'Pixel-only non-AI assessment must use medium confidence under original prompt')
    action, reason = 'pending', 'ai_visual_review_inconclusive'
    if parsed['needs_provenance']:
        reason = 'ai_visual_review_needs_external_evidence'
    elif parsed['assessment'] == 'possible_ai_edit':
        reason = 'ai_visual_review_edit_lineage_unresolved'
    elif parsed['assessment'] == 'likely_non_ai' and _cue_count(parsed, 'non_ai') >= 1:
        action, reason = 'retain_annotated', 'ai_visual_review_passed_screening'
    elif (parsed['assessment'] == 'likely_ai_generated' and parsed['confidence'] == 'high'
          and _cue_count(parsed, 'ai') >= 2):
        action, reason = 'exclude', 'ai_visual_review_high_confidence_generated'
    return {'asset_id': aid, 'screen_action': action, 'screen_reason': reason,
            'quality_reviewed': False, 'ai_decision': parsed['assessment'],
            'strict_blur_reviewed': False, 'decision_authority': AUTHORITY,
            'evidence_sha256': digest(evidence)}


def validate_review_package(package, manifest, missing_ids, prompt_binding, input_sha):
    require(package.get('schema') == 'corpus.assistant_visual_review.v1', 'Unknown assistant evidence schema')
    require(package.get('reviewer', {}).get('authority') == AUTHORITY,
            'Assistant evidence must not claim Qwen or human authority')
    require(package['reviewer'].get('qwen_inference_claimed') is False and
            package['reviewer'].get('human_review_claimed') is False and
            package['reviewer'].get('reviewer_type') == 'assistant_direct_visual_review',
            'Assistant evidence must explicitly disclaim Qwen and human authorship')
    require(package.get('user_instruction') == USER_INSTRUCTION, 'Explicit user instruction is not bound')
    require(package.get('prompt', {}).get('sha256') == prompt_binding['sha256'] and
            Path(package['prompt']['path']).resolve() == Path(prompt_binding['path']), 'Original prompt binding mismatch')
    require(manifest.get('schema') == 'corpus.assistant_visual_review_inputs.v1' and
            manifest.get('frozen_input_sha256') == input_sha and
            manifest.get('original_prompt_file_sha256') == prompt_binding['sha256'], 'Visual input manifest mismatch')
    images = manifest.get('images', []); reviews = package.get('reviews', [])
    image_map = {item['asset_id']: item for item in images}
    ids = [item.get('asset_id') for item in reviews]
    require(len(image_map) == len(images) == manifest.get('count'), 'Duplicate or missing visual input')
    require(len(ids) == len(set(ids)) and set(ids) == set(missing_ids) == set(image_map),
            'Assistant coverage must exactly match all unresolved requests, with no Qwen overlap')
    result = []
    for review in reviews:
        aid = review['asset_id']; image = image_map[aid]
        require(image.get('original_sha256') == aid and image.get('original_ref', {}).get('sha256') == aid and
                image.get('source_archive_and_member_sha256_verified') is True and image.get('crop') is False,
                'Original image binding is incomplete')
        observed = review.get('observed_image') or {}
        allowed = {(str(Path(image[key + '_path']).resolve()), image[key + '_sha256'])
                   for key in ('preview', 'full_resolution')}
        require((str(Path(observed.get('path', '')).resolve()), observed.get('sha256')) in allowed,
                'Observed image differs from the verified full-image rendering')
        evidence = {'schema': 'corpus.assistant_visual_evidence.v1', 'authority': AUTHORITY,
                    'reviewer': package['reviewer'], 'user_instruction': package['user_instruction'],
                    'prompt': prompt_binding, 'source': image, 'review': review,
                    'visual_input_manifest': package['visual_input_manifest'],
                    'qwen_inference_claimed': False, 'source_certification': False}
        result.append((normalize_assistant(review, evidence), evidence))
    return result


def verify_observed_pixels(raw, image, observed, expected_pixel_sha):
    """Reproduce the documented view transform and check the pixels actually seen."""
    from PIL import Image, ImageOps
    require(hashlib.sha256(raw).hexdigest() == image['asset_id'], 'Original asset bytes changed')
    bound(observed['path'], observed['sha256'])
    with Image.open(io.BytesIO(raw)) as original:
        original.load()
        rgba = ImageOps.exif_transpose(original).convert('RGBA')
        decoded = hashlib.sha256(b'RGBA' + struct.pack('>II', *rgba.size) + rgba.tobytes()).hexdigest()
        require(decoded == image['bound_decoded_rgba_geometry_sha256'] == expected_pixel_sha,
                'Original decoded pixel identity changed')
        if original.mode in ('RGBA', 'LA', 'P') or 'transparency' in original.info:
            expected = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
        else:
            expected = rgba.convert('RGB')
        if Path(observed['path']).resolve() == Path(image['preview_path']).resolve():
            expected.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        with Image.open(observed['path']) as actual:
            actual.load()
            require(actual.mode == 'RGB' and actual.size == expected.size and
                    actual.tobytes() == expected.tobytes(), 'Viewed pixels do not match the original image transform')


def write_history(area, rows, metadata):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from corpus_pipeline.dedup import decisions as legacy
    from corpus_pipeline.unified.policy import policy_hash
    area.mkdir(parents=True, exist_ok=True); path = area / 'screening.parquet'
    rows = sorted(rows, key=lambda row: row['asset_id'])
    if path.exists():
        require(pq.read_table(path).to_pylist() == rows, 'Existing history differs')
    else:
        pq.write_table(pa.Table.from_pylist(rows, schema=legacy.ARROW_SCHEMA), path, compression='zstd')
    marker = area / 'COMPLETE.json'
    immutable_json(marker, {'schema': legacy.SCHEMA, 'complete': True,
        'scope': 'exact_listed_evidence_subset', 'records': len(rows),
        'policy_sha256': policy_hash(), 'source_certification': False,
        'full_qwen_pass_claimed': False, 'metadata': metadata,
        'files': [{**bound(path), 'path': path.name}]})
    return marker


def validate_qwen_verification(report, success_count, original_bindings):
    require(report.get('complete') is True and report.get('error_count') == 0 and
            report.get('checked_success_receipts') == success_count,
            'Independent Qwen receipt verification is incomplete')
    for name, source in original_bindings.items():
        value = report.get('source_bindings', {}).get(name, {})
        require(value.get('sha256') == source['sha256'] and
                Path(value.get('path', '')).resolve() == Path(source['path']),
                'Independent Qwen verification belongs to changed inputs: ' + name)


def prepare_plan(run, area, expected_assistant_count):
    from corpus_pipeline.unified.workflow import verify_execution
    cfg = verify_execution(run / 'pipeline.json')
    rows = request_snapshot(run)
    missing = [row['asset_id'] for row in rows if row['request_state'] != 'ok']
    require(len(missing) == expected_assistant_count, 'Unresolved request count changed')
    identity = read(run / 'api_review/INPUT.json')
    require(len(rows) == identity['scope']['selected_assets'], 'API selection coverage changed')
    value = {'schema': SCHEMA, 'mode': 'plan_only', 'run_root': str(run),
        'base_execution': bound(run / 'EXECUTION.json'), 'base_config': bound(run / 'pipeline.json'),
        'original_api_input': bound(run / 'api_review/INPUT.json'),
        'original_api_complete': bound(run / 'api_review/COMPLETE.json'),
        'original_prompt': bound(cfg['screening']['prompt']), 'adapter': bound(__file__),
        'qwen_successful_requests': len(rows) - len(missing), 'assistant_assets_required': missing,
        'effective_authority_change': 'Unresolved Qwen requests replaced by independently sourced assistant visual evidence; successful Qwen results unchanged.',
        'frozen_assembly_overrides': {'review_root': None, 'history_additions': ['qwen_success_subset', 'assistant_visual_reviews']},
        'execution_and_api_identity_modified': False, 'science_outputs_written': False,
        'authorization': USER_INSTRUCTION}
    immutable_json(area / 'PLAN.json', value)
    return value


def apply(run, area, package_path, expected_assistant_count, verification_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from corpus_pipeline.unified import workflow, screening, release
    from corpus_pipeline.postprocess.api_screen import prepare_asset
    from corpus_pipeline.production.records import read_asset_bytes
    require(Path(workflow.__file__).resolve().is_relative_to(run / 'runtime'), 'Apply requires frozen runtime imports')
    with existing_writer_locks(run):
        require(not (run / 'STOP').exists(), 'Run STOP exists')
        cfg = workflow.verify_execution(run / 'pipeline.json')
        requests = request_snapshot(run); req = {row['asset_id']: row for row in requests}
        missing = {aid for aid, row in req.items() if row['request_state'] != 'ok'}
        require(len(missing) == expected_assistant_count, 'Unresolved request count changed')
        package_path = Path(package_path).resolve(); package = read(package_path)
        visual_binding = package['visual_input_manifest']
        bound(visual_binding['path'], visual_binding['sha256']); visual = read(visual_binding['path'])
        api_input_path = run / 'api_review/INPUT.json'; api_input = read(api_input_path)
        prompt = bound(cfg['screening']['prompt'])
        require(Path(api_input['prompt_path']).resolve() == Path(prompt['path']), 'Frozen prompt paths differ')
        reviewed = validate_review_package(package, visual, missing, prompt, sha(api_input_path))
        require(len(requests) == api_input['scope']['selected_assets'], 'API coverage differs from frozen selection')
        recon_sha = sha(run / 'reconciliation/COMPLETE.json')
        require(api_input['scope']['reconciliation_complete_sha256'] == recon_sha, 'API reconciliation changed')
        api_complete_path = run / 'api_review/COMPLETE.json'; api_complete = read(api_complete_path)
        require(api_complete['scope']['reconciliation_complete_sha256'] == recon_sha, 'API completion reconciliation changed')
        choices = [item for item in api_complete['artifacts'] if Path(item['path']).name == 'decisions.parquet']
        require(len(choices) == 1, 'Missing unique original API decisions binding')
        decision_source = run / 'api_review' / choices[0]['path']
        bound(decision_source, choices[0]['sha256'])
        verification_path = Path(verification_path).resolve()
        validate_qwen_verification(read(verification_path), len(requests) - len(missing),
            {'api_input': bound(api_input_path), 'api_complete': bound(api_complete_path),
             'decisions': bound(decision_source)})
        source_paths = [run / 'EXECUTION.json', run / 'pipeline.json', api_input_path,
                        api_complete_path, decision_source, Path(prompt['path']), package_path,
                        Path(visual_binding['path']), Path(__file__).resolve(), verification_path]
        original_bindings = [bound(path) for path in source_paths]
        revision = digest({'schema': SCHEMA, 'inputs': original_bindings,
                           'requests_sha256': digest(requests)})
        output = area / 'adoptions' / revision
        output.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for index, source in enumerate(source_paths):
            snapshots.append(snapshot_file(source, output / 'sources' / (str(index) + '_' + source.name)))
        source_decisions = Path(snapshots[4]['path'])
        qwen = []; seen = set()
        for batch in pq.ParquetFile(source_decisions).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                aid = row['asset_id']
                require(aid in req and aid not in seen, 'Unexpected/duplicate original API result')
                seen.add(aid)
                if aid not in missing:
                    qwen.append(normalized_qwen(row, req[aid]))
        require(seen == set(req) and len(qwen) + len(reviewed) == len(req), 'Mixed review coverage is incomplete')
        original_images = {item['asset_id']: item for item in visual['images']}
        with connect_ledger(run) as con:
            for normalized, evidence in reviewed:
                aid = normalized['asset_id']; image = original_images[aid]
                row = con.execute('SELECT input_sha,input_json FROM assets WHERE asset_id=?', (aid,)).fetchone()
                require(row is not None and row['input_sha'] == image['input_sha256'], 'Visual/ledger input mismatch')
                ledger_input = json.loads(row['input_json'])
                prepared = prepare_asset(ledger_input)
                require(prepared is not None and prepared[0]['original_ref'] == image['original_ref'],
                        'Original eligibility/reference checks failed')
                raw = read_asset_bytes(image['original_ref'], roots=run / 'source_roots.json')
                observed = evidence['review']['observed_image']
                verify_observed_pixels(raw, image, observed, ledger_input['pixel_sha256'])
        request_path = output / 'requests.parquet'
        if not request_path.exists(): pq.write_table(pa.Table.from_pylist(requests), request_path, compression='zstd')
        require(pq.read_table(request_path).to_pylist() == requests, 'Request evidence snapshot changed')
        evidence_path = output / 'ASSISTANT_EVIDENCE.json'
        immutable_json(evidence_path, {'schema': 'corpus.assistant_visual_evidence_bundle.v1',
                                      'reviews': [evidence for _, evidence in reviewed]})
        qhist = write_history(output / 'qwen_history', qwen, {'authority': 'qwen_visual_screening',
                             'original_decisions': snapshots[4], 'original_api_input': snapshots[2],
                             'requests': bound(request_path), 'normalizer_sha256': sha(screening.legacy.__file__)})
        ahist = write_history(output / 'assistant_history', [row for row, _ in reviewed],
                             {'authority': AUTHORITY, 'evidence': bound(evidence_path), 'adapter': bound(__file__)})
        histories = [*cfg['screening']['history_decision_manifests'], str(qhist), str(ahist)]
        adoption = {'schema': SCHEMA, 'revision': revision, 'base_execution': original_bindings[0],
            'base_config': original_bindings[1], 'original_inputs': original_bindings, 'snapshots': snapshots,
            'adapter': bound(__file__), 'assistant_evidence': bound(evidence_path),
            'independent_qwen_receipt_verification': bound(verification_path),
            'effective_history_manifests': [bound(path) for path in histories],
            'effective_authorities': {'successful_qwen_unchanged': len(qwen), AUTHORITY: len(reviewed)},
            'original_qwen_pass_complete': api_complete.get('screening_pass_complete'),
            'authorization': USER_INSTRUCTION, 'review_root_override': None,
            'effective_authority_changed': True, 'frozen_execution_modified': False,
            'api_input_or_results_modified': False, 'source_certification': False,
            'frozen_assembler_sha256': sha(screening.__file__),
            'frozen_receipt_sealer_sha256': sha(release.__file__)}
        adoption_path = output / 'ADOPTION.json'; immutable_json(adoption_path, adoption)
        rules = []
        for rule in cfg['screening']['quality_decisions']:
            value = read(rule) if isinstance(rule, str) else rule
            rules.extend(value if isinstance(value, list) else [value])
        result = screening.build_decisions(run / 'reconciliation', run / 'screening',
            review_root=None, quality_decisions=rules, history_decision_manifests=histories)
        require(result['complete'] is True and result.get('pending_infra') == 0,
                'Frozen screening assembler left missing or conflicting evidence')
        require([bound(path) for path in source_paths] == original_bindings and
                request_snapshot(run) == requests, 'Original API evidence changed during adoption')
        require(workflow.validated_receipt(run, 'reconcile') is not None, 'Missing validated reconciliation receipt')
        artifacts = workflow.finalized_files(run / 'screening')
        inputs = {str(run / 'EXECUTION.json'), str(adoption_path), str(evidence_path), str(request_path),
                  *(item['path'] for item in original_bindings), *(item['path'] for item in snapshots),
                  *(item['path'] for item in result['identity']['inputs'].values())}
        inputs.update(item['review']['observed_image']['path'] for _, item in reviewed)
        statistics = {key: result[key] for key in ('records', 'review_candidates', 'review_uncertain', 'counts')}
        statistics['evidence_authorities'] = adoption['effective_authorities']
        receipt = release.seal_receipt(run / 'receipts/screening.json', stage='screening',
            collections=workflow.collection_bindings(run),
            coverage={'unit': 'assets', 'expected': result['records'], 'processed': result['records'],
                      'missing': 0, 'pending_infra': 0}, inputs=sorted(inputs), artifacts=artifacts,
            dependencies={'reconcile': sha(run / 'receipts/reconcile.json')}, statistics=statistics,
            semantics={'scope': cfg['scope'], 'required_collections': cfg['required_collections'],
                       'task_direction_preserved': True, 'ordered_supervision_preserved': True,
                       'sscd_candidates_automatically_deleted': False, 'effective_authority_changed': True,
                       'evidence_adoption_schema': SCHEMA, 'adoption_sha256': sha(adoption_path),
                       'qwen_full_pass_claimed': False, 'source_certification': False})
        require(workflow.validated_receipt(run, 'screening') == receipt, 'Frozen stage validation failed')
        done = {'schema': SCHEMA, 'complete': True, 'adoption': bound(adoption_path),
                'screening_complete': bound(run / 'screening/COMPLETE.json'),
                'stage_receipt': bound(run / 'receipts/screening.json'),
                'effective_authorities': adoption['effective_authorities'], 'statistics': statistics,
                'frozen_stage_validation_passed': True, 'original_api_files_unchanged': True}
        immutable_json(output / 'COMPLETE.json', done)
        return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'apply'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--operations-dir', required=True)
    parser.add_argument('--reviews')
    parser.add_argument('--qwen-verification')
    parser.add_argument('--expected-assistant-count', type=int, default=10)
    args = parser.parse_args(argv); run = Path(args.run).resolve(); area = Path(args.operations_dir).resolve()
    require(area.is_relative_to(run / 'operations'), 'Adoption evidence must be in a separate operations directory')
    sys.path.insert(0, str(run / 'runtime'))
    if args.action == 'plan':
        value = prepare_plan(run, area, args.expected_assistant_count)
    else:
        require(args.reviews is not None, 'Apply requires completed visual review evidence')
        require(args.qwen_verification is not None, 'Apply requires complete independent Qwen receipt verification')
        value = apply(run, area, args.reviews, args.expected_assistant_count, args.qwen_verification)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
