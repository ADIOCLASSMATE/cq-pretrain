"""Frozen uncertain-review inputs and genuinely decoded image attachments.

Only this module's new manifest/attachment outputs are written. Source SQLite
uses a read-only consistent transaction; original TAR containers are read-only.
Source provenance and prior reviews are audit data, not blind-review prompts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import warnings

from PIL import Image, ImageFile, ImageOps, __version__ as PILLOW_VERSION

MAX_SOURCE_BYTES = 128 << 20
MAX_DECODED_PIXELS = 100_000_000
PENDING_REASON = 'visual_annotation_only_requires_confirmation'
FAILURE_REASON = 'api_review_failure_exhausted'
MANIFEST_SCHEMA = 'corpus.codex_review.candidates.v1'


class InputIntegrityError(ValueError):
    """Source identity, immutable scope, or bounded byte-range validation failed."""


class ImagePreparationError(ValueError):
    """The complete image could not be safely decoded and prepared."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _require(condition, code):
    if not condition:
        raise InputIntegrityError(code)


def _stat(path):
    info = Path(path).stat()
    return {'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _db_files(path):
    return {str(p): _stat(p) for p in (path, Path(str(path) + '-wal')) if p.exists()}


def _ref(item_id, value):
    _require(_sha(item_id) and isinstance(value, dict) and value.get('sha256') == item_id,
             'item_and_source_sha256_mismatch')
    path = value.get('path')
    _require(isinstance(path, str) and '\0' not in path and Path(path).is_absolute()
             and Path(path).suffix.lower() == '.tar', 'source_must_be_absolute_uncompressed_tar')
    offset, length = value.get('offset'), value.get('length')
    _require(type(offset) is int and offset >= 0 and type(length) is int
             and 0 < length <= MAX_SOURCE_BYTES, 'invalid_or_oversized_source_byte_range')
    return dict(value)


def _artifact_evidence(directory):
    return {name: {'path': str(directory / name), 'sha256': _file_sha(directory / name)}
            for name in ('INPUT.json', 'SUMMARY.json', 'COMPLETE.json') if (directory / name).exists()}


def export_candidates(source_db, output_jsonl) -> dict:
    """Stream a stable asset-ID ordered selection, with immutable input evidence.

    A completed production run's final_model_counts.json supplies the independently
    aggregated expected counts (70,462 + 30 for the authorized DCVLM run). Small
    fixtures without that report remain supported, using the same strict filter.
    The output and its .manifest.json sidecar are never overwritten.
    """
    source = Path(source_db).resolve(strict=True)
    output = Path(output_jsonl).absolute()
    sidecar = output.with_suffix('.manifest.json')
    _require(source != output and source != sidecar, 'output_must_not_replace_source_database')
    if output.exists() or sidecar.exists():
        raise FileExistsError('candidate_output_already_exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    before = _db_files(source)
    artifacts = _artifact_evidence(source.parent)
    expected = None
    expected_evidence = None
    final_counts = source.parent.parent / 'status_reports/final_model_counts.json'
    if final_counts.exists():
        final = json.loads(final_counts.read_text())
        _require(Path(final['database']).resolve() == source, 'final_count_report_database_mismatch')
        _require(final.get('screening_pass_complete') is True, 'source_screening_incomplete')
        _require(final.get('summary_sha256') == artifacts.get('SUMMARY.json', {}).get('sha256')
                 and final.get('complete_sha256') == artifacts.get('COMPLETE.json', {}).get('sha256'),
                 'final_count_report_artifact_binding_mismatch')
        expected = {'model_pending': final['model_recommendation']['pending']['count'],
                    'technical_failure': final['technical_failure_exhausted']['count']}
        expected_evidence = {'path': str(final_counts), 'sha256': _file_sha(final_counts)}
    if 'COMPLETE.json' in artifacts:
        _require(json.loads((source.parent / 'COMPLETE.json').read_text()).get('screening_pass_complete') is True,
                 'source_screening_not_complete')

    query = '''SELECT asset_id,input_sha,input_json,state,reason,decision_json FROM assets
      WHERE state='done' AND (reason=? OR
        (reason=? AND json_extract(decision_json,'$.model_recommendation')='pending'))
      ORDER BY asset_id COLLATE BINARY'''
    connection = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA busy_timeout=30000')
    connection.execute('BEGIN')
    fd, temporary_name = tempfile.mkstemp(prefix='.' + output.name + '.', suffix='.tmp', dir=output.parent)
    temporary = Path(temporary_name)
    counts, containers = Counter(), {}
    ids_hash, evidence_hash, bytes_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    count, previous_id, first_id = 0, None, None
    snapshot = datetime.now(timezone.utc).isoformat()
    try:
        with os.fdopen(fd, 'wb') as destination:
            for aid, input_sha, encoded, state, reason, previous in connection.execute(query, (FAILURE_REASON, PENDING_REASON)):
                _require(previous_id is None or aid > previous_id, 'candidate_ids_not_unique_and_ordered')
                record, prior = json.loads(encoded), json.loads(previous)
                _require(isinstance(record, dict) and record.get('asset_id') == aid
                         and _digest(record) == input_sha, 'source_input_digest_mismatch')
                _require(isinstance(prior, dict) and prior.get('input_sha256') == input_sha
                         and prior.get('reason') == reason and prior.get('decision') == 'pending'
                         and prior.get('detector_alarm_cleared') is False
                         and prior.get('source_certification') is False, 'prior_review_binding_or_policy_mismatch')
                if reason == PENDING_REASON:
                    _require(prior.get('model_recommendation') == 'pending'
                             and isinstance(prior.get('reviews', {}).get('primary'), dict), 'invalid_model_pending_row')
                    category = 'model_pending'
                else:
                    _require(reason == FAILURE_REASON and prior.get('model_recommendation') is None
                             and not prior.get('reviews'), 'technical_failure_contains_valid_review')
                    category = 'technical_failure'
                _require(record.get('global_status') == 'ai_suspect_pending_api'
                         and not any(record.get(k) for k in ('pixel_sha256_conflict', 'explicit_negative',
                                    'valid_protected', 'has_invalid_observation')), 'source_global_integrity_not_eligible')
                reference = _ref(aid, json.loads(record['original_ref_json']))
                provenance = json.loads(record['provenance_json'])
                _require(isinstance(provenance, dict) and prior.get('provenance_sha256') == _digest(provenance),
                         'source_provenance_digest_mismatch')
                tar_path = reference['path']
                if tar_path not in containers:
                    containers[tar_path] = _stat(tar_path)
                stats = containers[tar_path]
                _require(reference['offset'] + reference['length'] <= stats['size_bytes'], 'source_byte_range_outside_tar')
                item = {'schema_version': MANIFEST_SCHEMA, 'item_id': aid, 'asset_id': aid,
                        'queue_order': count, 'category': category, 'source_ref': reference,
                        'source_path': f"{tar_path}#offset={reference['offset']}&length={reference['length']}",
                        'source_tar_path': tar_path, 'source_bytes': reference['length'],
                        'source_mtime_ns': stats['mtime_ns'], 'source_container_bytes': stats['size_bytes'],
                        'source_input_sha256': input_sha, 'prior_review_sha256': _digest(prior),
                        'provenance': provenance, 'prior_review': prior}
                line = (_canonical(item) + '\n').encode()
                destination.write(line)
                bytes_hash.update(line)
                ids_hash.update((aid + '\n').encode())
                evidence_hash.update((_canonical([aid, input_sha, _digest(prior)]) + '\n').encode())
                counts[category] += 1
                first_id = first_id or aid
                previous_id = aid
                count += 1
            destination.flush()
            os.fsync(destination.fileno())
        connection.rollback()
        _require(before == _db_files(source), 'source_database_changed_during_export')
        _require(artifacts == _artifact_evidence(source.parent), 'source_artifacts_changed_during_export')
        if expected is not None:
            _require({key: counts[key] for key in expected} == expected, 'selected_scope_count_mismatch')
        report = {'schema_version': MANIFEST_SCHEMA, 'complete': True, 'source_database': str(source),
                  'source_database_files': before, 'source_artifacts': artifacts, 'snapshot_utc': snapshot,
                  'selection': {'state': 'done', 'model_pending_reason': PENDING_REASON,
                                'model_recommendation': 'pending', 'technical_failure_reason': FAILURE_REASON},
                  'ordering': 'asset_id ASCII ascending; zero-based queue_order', 'item_count': count,
                  'counts': dict(counts), 'expected_counts': expected, 'expected_counts_evidence': expected_evidence,
                  'first_item_id': first_id, 'last_item_id': previous_id,
                  'ordered_item_ids_sha256': ids_hash.hexdigest(), 'source_row_evidence_sha256': evidence_hash.hexdigest(),
                  'output': str(output), 'output_bytes': temporary.stat().st_size, 'sha256': bytes_hash.hexdigest(),
                  'source_container_count': len(containers), 'source_decoded_during_export': False,
                  'database_content_sha256_computed': False, 'inference_submitted': False}
        # Hard-link publication is atomic and refuses replacing another output.
        os.link(temporary, output)
        temporary.unlink()
        side_fd, side_name = tempfile.mkstemp(prefix='.' + sidecar.name + '.', suffix='.tmp', dir=sidecar.parent)
        try:
            with os.fdopen(side_fd, 'w') as stream:
                stream.write(_canonical(report) + '\n'); stream.flush(); os.fsync(stream.fileno())
            os.link(side_name, sidecar)
        finally:
            Path(side_name).unlink(missing_ok=True)
        return report
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def prepare_image(item, directory, index, max_side=2048, max_pixels=4194304) -> dict:
    """Verify TAR bytes, fully decode one frame, and write a metadata-free PNG.

    EXIF orientation and proportional downscaling are allowed. Never crop,
    enlarge, silently select frame zero, or relax the decoder's safety limits.
    """
    _require(type(index) is int and index >= 0, 'invalid_attachment_index')
    _require(type(max_side) is int and 0 < max_side <= 2048 and type(max_pixels) is int
             and 0 < max_pixels <= 4194304, 'attachment_geometry_exceeds_review_contract')
    aid = item.get('item_id')
    reference = _ref(aid, item.get('source_ref'))
    path = Path(reference['path'])
    try:
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            _require(reference['offset'] + reference['length'] <= before.st_size, 'source_byte_range_outside_tar')
            if 'source_mtime_ns' in item:
                _require(item['source_mtime_ns'] == before.st_mtime_ns, 'source_container_mtime_changed')
            if 'source_container_bytes' in item:
                _require(item['source_container_bytes'] == before.st_size, 'source_container_size_changed')
            if 'source_bytes' in item:
                _require(item['source_bytes'] == reference['length'], 'source_member_size_changed')
            stream.seek(reference['offset'])
            blob = stream.read(reference['length'])
            after = os.fstat(stream.fileno())
        _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'source_changed_during_read')
        _require(len(blob) == reference['length'] and hashlib.sha256(blob).hexdigest() == aid,
                 'source_sha256_or_length_mismatch')
    except OSError as error:
        raise InputIntegrityError('source_read_failed:' + type(error).__name__) from error
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ImagePreparationError('unsafe_decoder_configuration:LOAD_TRUNCATED_IMAGES')
    try:
        with warnings.catch_warnings(record=True) as decoder_warnings:
            warnings.simplefilter('always', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob)) as original:
                original_width, original_height = original.size
                if original_width * original_height > MAX_DECODED_PIXELS:
                    raise ImagePreparationError('decoded_pixel_limit_exceeded')
                frames = getattr(original, 'n_frames', 1)
                if frames != 1:
                    raise ImagePreparationError('multiframe_requires_explicit_adapter')
                source_format, source_mode = original.format, original.mode
                orientation = original.getexif().get(274, 1)
                metadata_keys = sorted(original.info)
                original.load()  # Mandatory even when no resize is needed.
                picture = ImageOps.exif_transpose(original)
                picture.load()
                oriented_width, oriented_height = picture.size
                has_alpha = picture.mode in ('RGBA', 'LA') or 'transparency' in original.info
                if has_alpha:
                    rgba = picture.convert('RGBA')
                    picture = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
                else:
                    picture = picture.convert('RGB')
                scale = min(1., max_side / picture.width, max_side / picture.height,
                            math.sqrt(max_pixels / (picture.width * picture.height)))
                dimensions = (max(1, int(picture.width * scale)), max(1, int(picture.height * scale)))
                if dimensions != picture.size:
                    picture = picture.resize(dimensions, Image.Resampling.LANCZOS)
                # A new raster strips ICC/EXIF/text metadata rather than raising
                # Pillow's text/ICC decompression limits to accommodate it.
                clean = Image.new('RGB', picture.size)
                clean.paste(picture)
                encoded = io.BytesIO()
                clean.save(encoded, format='PNG', compress_level=1)
                sent = encoded.getvalue()
        with Image.open(io.BytesIO(sent)) as verification:
            verification.load()
            if verification.size != dimensions or getattr(verification, 'n_frames', 1) != 1:
                raise ImagePreparationError('normalized_attachment_verification_failed')
    except ImagePreparationError:
        raise
    except Exception as error:
        raise ImagePreparationError('complete_image_decode_failed:' + type(error).__name__) from error

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'{index:02d}-{aid}.png'
    fd, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(sent); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    attachment = destination.stat()
    return {'item_id': aid, 'path': str(destination), 'source_sha256': aid,
            'sent_sha256': hashlib.sha256(sent).hexdigest(), 'decoded': True, 'crop': False,
            'decoded_width': original_width, 'decoded_height': original_height,
            'oriented_width': oriented_width, 'oriented_height': oriented_height,
            'width': dimensions[0], 'height': dimensions[1], 'size_bytes': attachment.st_size,
            'mtime_ns': attachment.st_mtime_ns, 'source_bytes': len(blob),
            'source_container_bytes': before.st_size, 'source_mtime_ns': before.st_mtime_ns,
            'source_ref': reference, 'decoder': 'Pillow', 'decoder_version': PILLOW_VERSION,
            'source_format': source_format, 'source_mode': source_mode, 'frame_count': frames,
            'exif_orientation': orientation, 'metadata_stripped': True, 'removed_metadata_keys': metadata_keys,
            'alpha_background': 'white' if has_alpha else None, 'media_type': 'image/png',
            'normalization': 'full_decode_exif_white_alpha_proportional_resize_metadata_free_png',
            'png_compress_level': 1, 'resample': 'LANCZOS', 'max_side': max_side, 'max_pixels': max_pixels,
            'max_source_bytes': MAX_SOURCE_BYTES, 'max_decoded_pixels': MAX_DECODED_PIXELS,
            'decoder_warnings': [type(w.message).__name__ for w in decoder_warnings]}
