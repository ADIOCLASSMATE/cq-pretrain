"""Stratified, reproducible, read-only extraction of actual DCVLM image bytes.

The source-domain groups are diagnostic strata, NEVER per-image authenticity
labels. Sampling is fixed before detector scores exist. Three PPS-selected
shards per source and random row positions bound I/O; this is not a corpus-
prevalence sample. One image is drawn uniformly from each sampled document.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Iterable

DEFAULT_ROOT = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z')
DEFAULT_OUTPUT = Path('corpus_pipeline/runs/detector_calibration_20261022')

DOMAIN_TARGETS = {'photo': 480, 'chart': 400, 'document': 400,
                  'rendering_screenshot': 400, 'mixed_web': 400, 'other_visual': 320}
CHART = set('chart2text chartgemma chartllama_dataset chartqa chartx dvqa figureqa plotqa scigraphqa simchart9k unichart vistext mavis_function'.split())
DOCUMENT = set('bigdocsbench docmatix docreason25k docstruct4m docvqa fintabnet sroie_2019 poie ocr_vqa postersum chinese_ocr casia iam hme100k mathwriting visualmrc tabmwp'.split())
RENDERING = set('clevr_math cosyn400k renderedtext synthdog synthtext websight multiui screen2words pixmo_docs parsynth_ocr_200k algopuzzlevqa'.split())
PHOTO = set('as100m as_v2 facecaption15m objects365 openimages_caption aokvqa coco_caption cocotext gqa inaturalist2018 internvl_sa1b_caption lvis_instruct_4v okvqa refcoco svit visdial visual7w vqav2 vsr ctw est_vqa lsvt mtwi rects st-vqa textcaps textocr textocr_gpt4v textvqa tallyqa tolokavqa v3det viquae kvqa'.split())
MIXED_WEB = set('datacomp_1b grit_caption mint_html_matteo mint_html_mehdi mint_html_vishaal mint_pdf multimodal_textbook omnicc_matteo omnicc_mehdi omnicc_vishaal wanjuan_matteo pixmo_cap pixmo_points'.split())


def domain_for(source: str) -> str:
    for domain, values in [('chart', CHART), ('document', DOCUMENT),
                           ('rendering_screenshot', RENDERING), ('photo', PHOTO), ('mixed_web', MIXED_WEB)]:
        if source in values:
            return domain
    if source.startswith('relaion-'):
        return 'mixed_web'
    return 'other_visual'


def stable_seed(seed: int, *parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps([seed, *parts], ensure_ascii=False).encode()).digest()[:8], 'big')


def split_for(image_sha256: str) -> str:
    # All copies of the same byte-identical image have one split, even if sources differ.
    return 'holdout' if stable_seed(20261022, 'locked_split', image_sha256) % 5 == 0 else 'calibration'


def weighted_shards(items: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Exponential race: PPS sampling without replacement using catalog rows."""
    ranked = [( -math.log(max(rng.random(), 1e-300)) / max(1, item['rows']), item['path'], item)
              for item in sorted(items, key=lambda x: x['path']) if item['rows'] > 0]
    return [item for _, _, item in sorted(ranked)[:count]]


def quota_plan(sources: dict[str, dict], target: int = 2400) -> dict[str, int]:
    groups = defaultdict(list)
    for source in sorted(sources):
        groups[domain_for(source)].append(source)
    quotas = {}
    for domain, names in sorted(groups.items()):
        budget = round(DOMAIN_TARGETS[domain] * target / 2400)
        if budget < len(names):
            raise ValueError(f'Target too small to cover all {domain} sources')
        base, extra = divmod(budget, len(names))
        for i, name in enumerate(names):
            quotas[name] = base + (i < extra)
    return quotas


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Path escapes declared dataset root: {relative}')
    return path


def extract_bytes(pool: Path, source_shard: str, ref: dict, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    path = contained(pool, source_shard)
    offset, length = ref['offset'], ref['length']
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or not 0 < length <= max_bytes:
        raise ValueError('Invalid or excessive image byte range')
    if offset + length > path.stat().st_size:
        raise ValueError('Image byte range exceeds source TAR size')
    with path.open('rb') as stream:
        stream.seek(offset)
        blob = stream.read(length)
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != length or not ref.get('sha256') or digest != ref['sha256']:
        raise ValueError('Source image byte length/SHA256 mismatch')
    return blob


def random_rows(path: Path, row_positions: Iterable[int]):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    requested = set(row_positions)
    start = 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        local = sorted(p - start for p in requested if start <= p < start + size)
        if local:
            table = parquet.read_row_group(group)
            for offset in local:
                yield start + offset, table.slice(offset, 1).to_pylist()[0]
        start += size


def dump_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sources', type=Path, default=Path('corpus_pipeline/config/sources.json'))
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--target', type=int, default=2400)
    parser.add_argument('--shards-per-source', type=int, default=3)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'samples.jsonl').exists():
        raise FileExistsError('Locked samples.jsonl already exists; choose a new output directory')
    images_dir = output / 'images'
    images_dir.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'COMPLETE.json').read_text())
    catalog_bytes = (root / 'catalog.json').read_bytes()
    if manifest.get('schema_version') != 'dcvlm.pretrain.v1' or manifest.get('phase') != 'complete':
        raise ValueError('Only complete native DCVLM v1 input is allowed')
    if complete.get('validation_passed') is not True or complete.get('catalog_sha256') != hashlib.sha256(catalog_bytes).hexdigest():
        raise ValueError('Input catalog is not validated by COMPLETE.json')
    if 't2i50' in str(root).lower():
        raise ValueError('Retired T2I50 input is forbidden')
    pool = Path(manifest['source_pool']).resolve()
    registry = json.loads(args.sources.read_text())
    all_sources = {x['source_id']: x for x in registry['sources'] if x.get('corpus') == 'dcvlm'}
    catalog = json.loads(catalog_bytes)
    by_source = defaultdict(list)
    for item in catalog:
        by_source[item['source']].append(item)
    missing = set(by_source) - set(all_sources)
    if missing:
        raise ValueError(f'Unregistered catalog sources: {sorted(missing)}')
    visual = {name: s for name, s in all_sources.items() if name in by_source and any(
        s.get('formats', {}).get(k, 0) for k in ('caption', 'visual_conversation', 'interleaved'))}
    quotas = quota_plan(visual, args.target)
    plan = {'seed': args.seed, 'target': args.target, 'shards_per_source': args.shards_per_source,
            'method': 'source/domain stratification; PPS shards without replacement; uniform random row positions; one uniform image per document; exact-byte SHA256 dedup',
            'domain_scope': 'heuristic source domain labels, not per-image content or authenticity truth',
            'row_candidate_budget': 'max(128, quota*12), capped at 2048 per selected shard set',
            'split': 'SHA256-keyed deterministic 80% calibration / 20% holdout; locked before detector scoring',
            'prevalence_estimation_valid': False, 'source_quotas': quotas,
            'source_root': str(root), 'source_pool': str(pool),
            'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
            'registry_sha256': hashlib.sha256(args.sources.read_bytes()).hexdigest()}
    dump_json(output / 'sampling_plan.json', plan)
    samples, associations, failures, source_stats = {}, [], [], []
    for ordinal, source in enumerate(sorted(visual), 1):
        rule, quota = visual[source], quotas[source]
        rng = random.Random(stable_seed(args.seed, source))
        shards = weighted_shards(by_source[source], args.shards_per_source, rng)
        total_rows = sum(s['rows'] for s in shards)
        positions = rng.sample(range(total_rows), min(total_rows, max(128, min(2048, quota * 12))))
        # Keep priorities random across shard/row boundaries; loading stays bounded to one row group.
        priority = {p: i for i, p in enumerate(positions)}
        candidates, begin = [], 0
        for item in shards:
            selected = [p - begin for p in positions if begin <= p < begin + item['rows']]
            path = contained(root, item['path'])
            if path.stat().st_size != item['bytes']:
                raise ValueError(f'Catalog byte size mismatch: {path}')
            for row_index, row in random_rows(path, selected):
                if row['source'] != source or row['source_shard'] != item['source_shard']:
                    raise ValueError('Parquet row/catalog source mismatch')
                if row.get('images'):
                    # Choose image independent of content, captions, and any model score.
                    image_index = random.Random(stable_seed(args.seed, source, row['sample_id'], 'image')).randrange(len(row['images']))
                    candidates.append((priority[begin + row_index], row_index, item['path'], image_index, row))
            begin += item['rows']
        accepted, duplicates, inspected = 0, 0, 0
        for _, row_index, parquet_relative, image_index, row in sorted(candidates, key=lambda c: c[0]):
            if accepted >= quota:
                break
            inspected += 1
            ref = row['images'][image_index]
            digest = ref['sha256']
            association = {'image_sha256': digest, 'source': source, 'sample_id': row['sample_id'],
                           'image_index': image_index, 'parquet_path': str(root / parquet_relative),
                           'parquet_row_index': row_index, 'source_shard': row['source_shard'],
                           'source_key': row.get('source_key'), 'row_ordinal': row.get('row_ordinal'),
                           'image_ref': ref, 'format': row.get('format'), 'modality': row.get('modality'),
                           'segments': row.get('segments'), 'source_image_count': len(row['images']),
                           'annotation_file': row.get('annotation_file'),
                           'caption_origin': rule.get('caption_origin', 'unknown')}
            associations.append(association)
            if digest in samples:
                duplicates += 1
                samples[digest]['sampled_association_count'] += 1
                if source not in samples[digest]['associated_sources']:
                    samples[digest]['associated_sources'].append(source)
                continue
            try:
                blob = extract_bytes(pool, row['source_shard'], ref)
                from PIL import Image
                with Image.open(io.BytesIO(blob)) as image:
                    image.load()
                    size, fmt, mode = image.size, image.format, image.mode
                extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif',
                             'TIFF': '.tiff', 'BMP': '.bmp'}.get(fmt, '.img')
                destination = images_dir / (digest + extension)
                if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError('Existing extracted image SHA256 mismatch')
                if not destination.exists():
                    destination.write_bytes(blob)
                samples[digest] = {
                    'image_sha256': digest, 'image_path': str(destination),
                    'source': source, 'domain': domain_for(source),
                    'domain_label_scope': 'source_heuristic_not_per_image_truth',
                    'source_image_origin': rule.get('image_origin', 'unknown'),
                    'source_evidence_level': rule.get('evidence_level'),
                    'source_evidence': rule.get('evidence', []),
                    'authenticity_ground_truth': None,
                    'ai_edited_from_real_ground_truth': None,
                    'calibration_split': split_for(digest),
                    'sample_id': row['sample_id'], 'image_index': image_index,
                    'tar_path': str(contained(pool, row['source_shard'])),
                    'tar_member': ref.get('member'), 'tar_offset': ref['offset'], 'tar_length': ref['length'],
                    'pixel_sha256': ref.get('pixel_sha256'), 'source_url': ref.get('source_url'),
                    'decoded_width': size[0], 'decoded_height': size[1], 'decoded_mode': mode, 'decoded_format': fmt,
                    'caption_origin': rule.get('caption_origin', 'unknown'),
                    'segments': row.get('segments'),
                    'parquet_path': str(root / parquet_relative), 'parquet_row_index': row_index,
                    'source_key': row.get('source_key'), 'format': row.get('format'),
                    'source_image_count': len(row['images']),
                    'associated_sources': [source], 'sampled_association_count': 1,
                    'sampling_seed': args.seed,
                }
                accepted += 1
            except Exception as exc:
                failures.append({'source': source, 'sample_id': row['sample_id'], 'image_sha256': digest,
                                 'error_type': type(exc).__name__, 'message': str(exc)})
        stats = {'source': source, 'domain': domain_for(source), 'quota': quota,
                 'unique_images': accepted, 'duplicate_images': duplicates,
                 'sampled_row_candidates': len(positions), 'visual_candidate_rows': len(candidates),
                 'inspected_image_candidates': inspected, 'selected_shards': [s['path'] for s in shards],
                 'available_shards': len(by_source[source]), 'source_image_origin': rule.get('image_origin', 'unknown')}
        source_stats.append(stats)
        print(json.dumps({'progress': f'{ordinal}/{len(visual)}', **stats}, ensure_ascii=False), flush=True)
    for filename, records in [('samples.jsonl', samples.values()), ('sample_associations.jsonl', associations), ('sampling_failures.jsonl', failures)]:
        with (output / filename).open('w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = {
        **{k: v for k, v in plan.items() if k != 'source_quotas'},
        'catalog_sources': len(by_source), 'visual_sources': len(visual),
        'text_only_sources_skipped': sorted(set(by_source) - set(visual)),
        'unique_images': len(samples), 'sampled_associations': len(associations),
        'duplicate_associations': sum(s['duplicate_images'] for s in source_stats),
        'failed_images': len(failures), 'image_bytes': sum(s['tar_length'] for s in samples.values()),
        'domain_counts': dict(Counter(s['domain'] for s in samples.values())),
        'split_counts': dict(Counter(s['calibration_split'] for s in samples.values())),
        'source_origin_counts': dict(Counter(s['source_image_origin'] for s in samples.values())),
        'sources': source_stats,
        'registered_ai_generated_sources': [s for s, x in all_sources.items() if x.get('image_origin') == 'ai_generated'],
        'registered_ai_edited_sources': [s for s, x in all_sources.items() if x.get('image_origin') == 'ai_edited'],
        'missing_confirmed_classes': ['AI-generated image positive ground truth', 'AI-edited image with verified real-source lineage'],
        'class_coverage_limitation': 'Registry has no source declared AI-generated/AI-edited; mixed/unknown pools may still contain such images. This sample does not prove their absence. Detector recall or an edit-preserving threshold cannot be calibrated without verified positive/edit labels.',
        'source_ground_truth_warning': 'Source provenance is weak evidence, not a per-image authenticity ground truth. mixed_web is never labeled as real.',
        'samples_sha256': hashlib.sha256((output / 'samples.jsonl').read_bytes()).hexdigest(),
        'sampling_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dump_json(output / 'sampling_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('unique_images', 'failed_images', 'domain_counts', 'source_origin_counts', 'split_counts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
