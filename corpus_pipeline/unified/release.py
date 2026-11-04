"""Generic receipt-gated publication, independent of hand-written run audits.

Stages bind exact input/output bytes, policy, collection snapshots and coverage.
Final publication validates that graph, rechecks source masks and publishes an
immutable release version before atomically moving CURRENT.json.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import time

import numpy as np

from corpus_pipeline.postprocess.reconcile import atomic_json, digest, file_sha, locked, require
from corpus_pipeline.dedup.source_view import CODES, iter_native_batches, verify_policy_overlay
from .policy import config, policy_hash
from .config import collection_scope

SCHEMA = 'corpus.unified.stage_receipt.v1'
RELEASE_SCHEMA = 'corpus.unified.release.v1'
REQUIRED_STAGES = ('normalize', 'frontpass', 'reconcile', 'screening', 'sscd_catalog',
                   'sscd_search', 'assets', 'graph', 'records', 'source_view')


def artifact(path):
    path = Path(path).resolve(strict=True); before = path.stat()
    result = {'path': str(path), 'sha256': file_sha(path), 'size_bytes': before.st_size}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Artifact changed during hashing')
    return result


def _artifacts(values):
    result = []
    for value in values:
        if isinstance(value, (str, Path)):
            result.append(artifact(value))
        else:
            current = artifact(value['path'])
            require(current['sha256'] == value['sha256'] and current['size_bytes'] == value['size_bytes'], 'Artifact binding changed')
            result.append({**value, **current})
    require(len({x['path'] for x in result}) == len(result), 'Duplicate artifact path')
    return result


def _coverage(value):
    require(value.get('unit') in {'records', 'assets', 'image_occurrences', 'vector_pairs', 'vectors'}, 'Unknown coverage unit')
    for key in ['expected', 'processed', 'missing', 'pending_infra']:
        require(type(value.get(key)) is int and value[key] >= 0, 'Coverage requires nonnegative integer ' + key)
    require(value['expected'] == value['processed'] and value['missing'] == value['pending_infra'] == 0,
            'Stage coverage is incomplete or infrastructure results are missing')


def seal_receipt(path, *, stage, collections, coverage, inputs=(), artifacts=(), dependencies=None,
                 statistics=None, semantics=None, reader=None, policy_config=None):
    """Seal a successful stage; callers must supply actual measured coverage/counts."""
    _coverage(coverage)
    require(bool(collections) and all(isinstance(k, str) and k and re.fullmatch('[0-9a-f]{64}', v)
                                     for k, v in collections.items()), 'Invalid collection snapshot identities')
    semantics = {'scope': 'validation', **(semantics or {})}
    require(semantics['scope'] in {'validation', 'full'}, 'Unknown publication scope')
    receipt = {'schema': SCHEMA, 'stage': stage, 'complete': True,
        'policy_sha256': policy_hash(policy_config), 'collections': dict(sorted(collections.items())),
        'coverage': coverage, 'inputs': _artifacts(inputs), 'artifacts': _artifacts(artifacts),
        'dependencies': dependencies or {}, 'statistics': statistics or {}, 'semantics': semantics}
    if reader is not None: receipt['reader'] = reader
    require(receipt['artifacts'], 'A completed stage needs a bound output artifact')
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(json.loads(path.read_text()) == receipt, 'Immutable receipt already exists with different inputs')
    else: atomic_json(path, receipt)
    return receipt


def _counts(value):
    require(set(value) == {'keep', 'exclude', 'quarantine'}, 'Selection counts must include keep, exclude, quarantine')
    require(all(type(x) is int and x >= 0 for x in value.values()), 'Invalid selection counts')
    return Counter(value)


def _verify_source_view(path, expected_records, expected_counts):
    path = Path(path).resolve(strict=True)
    complete = json.loads((path/'COMPLETE.json').read_text()); manifest = json.loads((path/'manifest.json').read_text())
    require(complete.get('complete') is True and complete['manifest_sha256'] == file_sha(path/'manifest.json'), 'Source view incomplete or changed')
    require(complete['records'] == manifest['records'] == expected_records, 'Source-view row coverage differs')
    require(manifest['codes'] == CODES and manifest['original_records_modified'] is False,
            'Source view changed native data or uses unknown mask codes')
    require(manifest['counts'] == complete['counts'] == {**expected_counts, 'not_indexed': 0}, 'Source-view counts differ')
    totals = Counter(); seen = set(); rows = 0
    for item in manifest['files']:
        require(item['source_path'] not in seen, 'Duplicate source file in view'); seen.add(item['source_path'])
        target = (path/item['mask_path']).resolve()
        require(target.is_relative_to(path), 'Mask path escapes source view')
        require(target.stat().st_size == item['rows'] == item['mask_bytes'] and file_sha(target) == item['mask_sha256'], 'Selection mask changed')
        mask = np.fromfile(target, dtype=np.uint8)
        verify_policy_overlay(path,item)
        require(np.all(mask <= 2), 'Unindexed or invalid selection mask')
        counts = {key: int(np.count_nonzero(mask == code)) for key, code in CODES.items()}
        require(counts == item['counts'], 'Selection mask counts disagree')
        totals.update(counts); rows += len(mask)
    require(rows == expected_records == manifest['mask_bytes'] == manifest['native_rows'], 'Source masks do not cover every record')
    require(len(seen) == manifest['source_files'] and dict(totals) == manifest['counts'], 'Source-mask aggregate coverage differs')
    return {'kind': 'native_source_view', 'path': str(path), 'complete': artifact(path/'COMPLETE.json'),
            'manifest': artifact(path/'manifest.json'), 'source_files': len(seen), 'masks_verified': len(seen)}


def _validate_stage_graph(stages, bindings):
    visiting = set(); done = set()
    def visit(name):
        require(name not in visiting, 'Stage dependency cycle')
        if name in done: return
        visiting.add(name)
        for parent, expected in stages[name]['dependencies'].items():
            require(parent in stages and expected == bindings[parent]['sha256'], 'Stale or missing stage dependency')
            visit(parent)
        visiting.remove(name); done.add(name)
    for name in stages: visit(name)


def _verify_native_source_lineage(root, manifest, complete, artifacts):
    """Validate the compactor's pinned input proof before excluding its copy.

    A nested native COMPLETE is not another published snapshot only when the
    main manifest explicitly binds all four copied input documents. Merely
    naming a directory ``lineage`` or nesting another dataset is insufficient.
    """
    lineage = manifest['source_lineage']
    require(isinstance(lineage, dict) and lineage.get('contract') == 'complete_pinned_native_index.v1'
            and manifest.get('enumeration_scope') == 'derived_full_pinned_source_metadata'
            and manifest.get('catalog_contract') == complete.get('catalog_contract') == 'mixed_source_tar.v1'
            and manifest.get('source_files_modified') is False
            and manifest.get('original_row_fields_modified') is False
            and manifest.get('image_payloads_copied') is False,
            'Native source lineage contract is invalid')
    names = {'PLAN.json', 'COMPLETE.json', 'manifest.json', 'catalog.json'}
    bindings = lineage.get('artifacts', {})
    require(isinstance(bindings, dict) and set(bindings) == names, 'Native source lineage metadata is incomplete')
    documents = {}; paths = {}
    for name in sorted(names):
        binding = bindings[name]
        require(isinstance(binding, dict) and binding.get('path') == 'lineage/' + name,
                'Native source lineage path is not the declared local copy')
        target = (root / binding['path']).resolve(strict=True)
        require(target.is_relative_to(root) and target != root / name
                and target.stat().st_size == binding.get('bytes')
                and file_sha(target) == binding.get('sha256'), 'Native source lineage bytes changed')
        require(str(target) not in artifacts or artifacts[str(target)] == binding['sha256'],
                'Native source lineage artifact binding changed')
        documents[name] = json.loads(target.read_text()); paths[name] = target
    source_complete = documents['COMPLETE.json']; source_manifest = documents['manifest.json']
    source_plan = documents['PLAN.json']; source_catalog = documents['catalog.json']
    require(source_complete.get('schema_version') == source_manifest.get('schema_version') == 'corpus.native.v2'
            and source_complete.get('complete') is True and source_complete.get('validation_passed') is True
            and source_complete.get('scope') == source_manifest.get('scope') == 'full_pinned_upstream_release',
            'Native source lineage is not a fully verified pinned index')
    for name, key in [('PLAN.json', 'plan_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('catalog.json', 'catalog_sha256')]:
        require(source_complete.get(key) == bindings[name]['sha256'], 'Native source lineage receipt binding differs')
    require(source_manifest.get('plan_sha256') == bindings['PLAN.json']['sha256'],
            'Native source lineage manifest plan binding differs')
    validation = source_manifest.get('source_validation', {})
    require(validation.get('complete') is True
            and validation == lineage.get('raw_source_validation') == manifest.get('source_validation')
            and validation.get('repo') == source_plan.get('repo')
            and validation.get('revision') == source_plan.get('revision')
            and validation.get('files') == source_plan.get('source_total_files')
            and validation.get('bytes') == source_plan.get('source_total_bytes')
            and source_manifest.get('source_inventory_sha256') == source_plan.get('source_inventory_sha256')
            == manifest.get('source_inventory_sha256'), 'Native source lineage upstream identity differs')
    require(isinstance(source_catalog, list) and len(source_catalog) > 0,
            'Native source lineage input catalog is empty')
    source_totals = Counter(); seen = set()
    for entry in source_catalog:
        require(entry['path'] not in seen and type(entry['rows']) is int and entry['rows'] > 0,
                'Native source lineage input catalog coverage is invalid')
        seen.add(entry['path']); source_totals[(entry['collection'], entry['source'])] += entry['rows']
    declared = lineage.get('per_source_rows', [])
    declared_totals = {(entry['collection'], entry['source']): entry['rows'] for entry in declared}
    require(len(declared) == len(declared_totals) and dict(source_totals) == declared_totals
            and lineage.get('input_files') == lineage.get('input_files_covered_once') == len(source_catalog),
            'Native source lineage exact-once source coverage differs')
    statistics = source_manifest.get('statistics', {})
    require(source_complete.get('statistics') == statistics
            and sum(source_totals.values()) == statistics.get('samples') == complete['statistics']['samples']
            and set(source_manifest.get('collections', [])) == {key[0] for key in source_totals}
            == set(manifest['collections'])
            and all(manifest['statistics'].get(key) == value for key, value in statistics.items()),
            'Native source lineage row or collection coverage differs')
    return paths['COMPLETE.json']


def _verify_full_native_snapshot(stage, collections):
    """A wrapper scope cannot promote a selected or bounded native snapshot."""
    artifacts = {item['path']: item['sha256'] for item in stage['artifacts']}
    candidates = []
    for path in artifacts:
        if Path(path).name != 'COMPLETE.json': continue
        value = json.loads(Path(path).read_text())
        if value.get('schema_version') == 'corpus.native.v2': candidates.append((Path(path), value))
    lineage_copies = set()
    for path, complete in candidates:
        manifest_path = path.parent / 'manifest.json'
        if str(manifest_path) not in artifacts:
            continue
        require(artifacts[str(manifest_path)] == complete.get('manifest_sha256') == file_sha(manifest_path),
                'Native snapshot metadata is not bound')
        manifest = json.loads(manifest_path.read_text())
        if 'source_lineage' in manifest:
            lineage_copies.add(_verify_native_source_lineage(path.parent, manifest, complete, artifacts))
    candidates = [(path, value) for path, value in candidates if path not in lineage_copies]
    require(len(candidates) == 1, 'Full publication requires one bound native v2 snapshot')
    path, complete = candidates[0]
    require(complete.get('complete') is True and complete.get('scope') == 'full_declared_inputs',
            'Bounded or selected native snapshots cannot publish as full')
    manifest_path = path.parent/'manifest.json'; catalog_path = path.parent/'catalog.json'
    for target, key in [(manifest_path, 'manifest_sha256'), (catalog_path, 'catalog_sha256')]:
        require(artifacts.get(str(target)) == complete[key] == file_sha(target), 'Native snapshot metadata is not bound')
    manifest = json.loads(manifest_path.read_text()); catalog = json.loads(catalog_path.read_text())
    require(manifest.get('schema_version') == 'corpus.native.v2' and manifest.get('scope') == 'full_declared_inputs',
            'Native manifest scope differs from full snapshot')
    require(set(manifest['collections']) == {item['collection'] for item in catalog} == set(collections),
            'Native snapshot collection coverage differs')
    actual = {name: digest({'snapshot': artifacts[str(path)], 'collection': name,
                           'files': [item for item in catalog if item['collection'] == name]})
              for name in collections}
    require(actual == collections, 'Collection identities do not bind the full native snapshot')
    require(complete['statistics']['samples'] == manifest['statistics']['samples'] == stage['coverage']['processed'],
            'Native snapshot row coverage differs')


def finalize(run_root, stage_receipts, policy_config=None, expected_collections=None,
             required_stages=REQUIRED_STAGES, required_collections=None):
    root = Path(run_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    require(expected_collections and set(required_stages).issubset(stage_receipts), 'Missing collection identities or required stages')
    stages = {}; bindings = {}
    for name, path in sorted(stage_receipts.items()):
        binding = artifact(path); value = json.loads(Path(binding['path']).read_text())
        require(value.get('schema') == SCHEMA and value.get('complete') is True and value['stage'] == name, 'Invalid stage completion receipt')
        require(value['policy_sha256'] == policy_hash(policy_config), 'Stage uses another screening policy')
        require(value['collections'] == expected_collections, 'Stage collection snapshot coverage differs')
        _coverage(value['coverage'])
        require(value.get('artifacts'), 'Stage has no output artifacts')
        _artifacts(value.get('inputs', [])); _artifacts(value['artifacts'])
        require(value.get('semantics', {}).get('sscd_candidates_automatically_deleted', False) is False,
                'SSCD candidates cannot silently become confirmed duplicates')
        stages[name] = value; bindings[name] = binding
    _validate_stage_graph(stages, bindings)
    scopes = {value.get('semantics', {}).get('scope') for value in stages.values()}
    require(len(scopes) == 1 and next(iter(scopes)) in {'validation', 'full'}, 'Stages have missing or inconsistent publication scope')
    scope = next(iter(scopes))
    default_collections = ['dcvlm', 'genund'] if scope == 'full' else sorted(expected_collections)
    declared = collection_scope(default_collections if required_collections is None else required_collections)
    require(set(expected_collections) == set(declared), 'Publication snapshots differ from required_collections')
    for stage in stages.values():
        stage_collections = collection_scope(stage.get('semantics', {}).get('required_collections', default_collections))
        require(stage_collections == declared, 'Stage required_collections differ from publication scope')
    if scope == 'full':
        require(set(REQUIRED_STAGES).issubset(stages), 'Full publication requires every production stage')
        _verify_full_native_snapshot(stages['normalize'], expected_collections)
    require('records' in stages and 'source_view' in stages, 'Publication requires records and source-view receipts')
    records = stages['records']; statistics = records['statistics']; count = statistics['records']
    require(records['coverage']['unit'] == 'records' and count == records['coverage']['processed'], 'Record coverage differs')
    for name, stage in stages.items():
        if stage['coverage']['unit'] == 'records':
            require(stage['coverage']['expected'] == count, 'Record coverage differs across stages: ' + name)
    require(stages['source_view']['coverage']['unit'] == 'records', 'Source-view coverage must count records')
    counts = dict(_counts(statistics['counts'])); require(sum(counts.values()) == count, 'Record decisions do not sum to coverage')
    per_collection = statistics['per_collection']; require(set(per_collection) == set(expected_collections), 'Record collection coverage differs')
    aggregate = Counter(); total = 0
    for item in per_collection.values():
        local = _counts(item['counts']); require(sum(local.values()) == item['records'], 'Collection decisions do not sum to coverage')
        aggregate.update(local); total += item['records']
    require(dict(aggregate) == counts and total == count, 'Joint collection statistics differ from global coverage')
    semantics = records.get('semantics', {})
    require(semantics.get('task_direction_preserved') is True and semantics.get('ordered_supervision_preserved') is True,
            'Record receipt must certify preservation of task direction and ordered supervision')
    require(stages['source_view']['dependencies'].get('records') == bindings['records']['sha256'], 'Source view is not bound to final record decisions')
    descriptor = stages['source_view'].get('reader', {})
    require(descriptor.get('kind') == 'native_source_view', 'Unsupported release reader')
    reader = _verify_source_view(descriptor['path'], count, counts)
    bound_paths = {x['path']: x['sha256'] for x in stages['source_view']['artifacts']}
    for key in ['complete', 'manifest']:
        item = reader[key]; require(bound_paths.get(item['path']) == item['sha256'], 'Reader manifest is not a stage artifact')
    if descriptor.get('asset_supervision_view'):
        from corpus_pipeline.dedup.asset_supervision_view import _read_manifest
        view_root,view_manifest=_read_manifest(descriptor['asset_supervision_view'])
        source_manifest=json.loads((Path(reader['path'])/'manifest.json').read_text())
        require(view_manifest['input_candidate_manifest_sha256']==source_manifest['input_candidate_manifest_sha256'],
                'Asset and native views refer to different candidate indexes')
        require(view_manifest['counts']['supervisions']==counts['keep'],
                'Asset supervision coverage differs from selected native records')
        view={'kind':'asset_supervision_view','path':str(view_root),
              'counts':view_manifest['counts'],
              'complete':artifact(view_root/'COMPLETE.json'),
              'manifest':artifact(view_root/'manifest.json')}
        for key in ('complete','manifest'):
            item=view[key]
            require(bound_paths.get(item['path'])==item['sha256'],'Unbound asset supervision view')
        reader['asset_supervision_view']=view
    identity = {'schema': RELEASE_SCHEMA, 'scope': scope, 'policy': config(policy_config), 'policy_sha256': policy_hash(policy_config),
                'collections': expected_collections, 'required_collections': declared,
                'stages': bindings, 'publisher_code_sha256': file_sha(__file__)}
    version = digest(identity); destination = root/'releases'/version
    with locked(root/'publication'):
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/'COMPLETE.json').exists():
            done = json.loads((destination/'COMPLETE.json').read_text())
            require(done['manifest_sha256'] == file_sha(destination/'manifest.json'), 'Immutable release manifest changed')
            manifest = json.loads((destination/'manifest.json').read_text())
            require(manifest['identity'] == identity, 'Immutable release identity changed')
        else:
            manifest = {'schema': RELEASE_SCHEMA, 'complete': True, 'scope': scope, 'identity': identity,
                'required_collections': declared,
                'records': count, 'counts': counts, 'per_collection': per_collection, 'reader': reader,
                'statistics': {name: row['statistics'] for name, row in stages.items()},
                'review_pending': statistics.get('review_pending', 0),
                'sscd_candidates_automatically_deleted': False, 'source_certification': False,
                'original_records_modified': False, 'created_at': time.time()}
            atomic_json(destination/'manifest.json', manifest)
            done = {'schema': RELEASE_SCHEMA, 'complete': True, 'scope': scope, 'release_id': version,
                    'required_collections': declared,
                    'manifest_sha256': file_sha(destination/'manifest.json'), 'records': count, 'counts': counts}
            atomic_json(destination/'COMPLETE.json', done)
        pointer = {'schema': RELEASE_SCHEMA, 'scope': scope, 'release_id': version, 'root': str(destination),
                   'required_collections': declared,
                   'complete': artifact(destination/'COMPLETE.json'), 'manifest': artifact(destination/'manifest.json')}
        atomic_json(root/'CURRENT.json', pointer)
    return {**done, 'root': str(destination), 'current': str(root/'CURRENT.json')}


def read_manifest(path):
    path = Path(path).resolve(strict=True)
    if path.is_dir(): path = path/'CURRENT.json' if (path/'CURRENT.json').exists() else path/'COMPLETE.json'
    value = json.loads(path.read_text())
    if 'root' in value:
        _artifacts([value['complete'], value['manifest']]); path = Path(value['root'])/'COMPLETE.json'; value = json.loads(path.read_text())
    require(value.get('complete') is True, 'Release incomplete')
    manifest_path = path.parent/'manifest.json'; require(file_sha(manifest_path) == value['manifest_sha256'], 'Release manifest changed')
    return json.loads(manifest_path.read_text())


def iter_batches(release, batch_size=8192, epoch_seed=None, *, worker_id=0, num_workers=1):
    manifest = read_manifest(release); reader = manifest['reader']
    _artifacts([reader['complete'], reader['manifest']])
    yield from iter_native_batches(reader['path'], batch_size, epoch_seed, worker_id=worker_id, num_workers=num_workers)
