"""SSCD feature cache and transactional historical/current-batch copy search.

Exact bytes/pixels share an asset. SSCD similarity only creates review edges:
different text, crops or edits must never silently share their supervision.
This catalog records checked image identity, not training eligibility.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import struct
import time

import numpy as np
from PIL import Image, ImageOps

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models/sscd/sscd_disc_mixup.torchscript.pt'
PREPROCESS = 'exif_transpose_rgba_white_rgb_pillow_bilinear_320x320_imagenet_norm_v1'
MODEL_SHA256 = '9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_image_asset(asset):
    """Read exactly one original asset; return validated metadata and model input."""
    loader = asset.get('blob_loader')
    if loader is not None:
        blob = loader()
    elif asset.get('image_path'):
        blob = Path(asset['image_path']).read_bytes()
    else:
        raise ValueError('image_path or blob_loader required')
    if not isinstance(blob, bytes) or len(blob) > 128 * 1024 * 1024:
        raise ValueError('invalid or excessive image bytes')
    sha = hashlib.sha256(blob).hexdigest()
    expected = asset.get('sha256', asset.get('image_sha256'))
    if not expected or sha != expected:
        raise ValueError('image SHA256 mismatch')
    with Image.open(io.BytesIO(blob)) as opened:
        if opened.width * opened.height > 100_000_000:
            raise ValueError('image pixel limit exceeded')
        if getattr(opened, 'n_frames', 1) != 1:
            raise ValueError('multi-frame image requires a frame-specific asset adapter')
        original_size = opened.size
        if min(original_size) < 128:
            raise ValueError('original image short edge is below 128')
        for key, value in zip(('width', 'height'), original_size):
            if asset.get(key) is not None and asset[key] != value:
                raise ValueError('declared original dimensions mismatch')
        rgba = ImageOps.exif_transpose(opened).convert('RGBA')
        rgba.load()
    pixel_sha = hashlib.sha256(b'RGBA' + struct.pack('>II', *rgba.size) + rgba.tobytes()).hexdigest()
    image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    tensor = np.asarray(image.resize((320, 320), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = ((tensor - np.array([.485, .456, .406], np.float32)) /
              np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy()
    metadata = {k: v for k, v in asset.items() if k not in ('blob_loader', 'bytes', 'embedding')}
    metadata.update(sha256=sha, pixel_sha256=pixel_sha, width=original_size[0], height=original_size[1],
                    oriented_width=image.width, oriented_height=image.height)
    metadata.setdefault('asset_id', sha)
    return metadata, tensor


class Encoder:
    """CPU inference is intentional on the control host; no NPU auto-import."""
    def __init__(self, model_path=DEFAULT_MODEL, threads=8):
        os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
        import torch
        self.torch = torch
        if file_sha(model_path) != MODEL_SHA256:
            raise ValueError('Unexpected SSCD checkpoint; register a new model identity before reuse')
        torch.set_num_threads(threads)
        self.model = torch.jit.load(str(model_path), map_location='cpu').eval()
        self.identity = {'model': 'sscd_disc_mixup', 'weight_sha256': MODEL_SHA256,
                         'preprocess': PREPROCESS, 'dimensions': 512,
                         'dtype': 'float32', 'device': 'cpu', 'torch': torch.__version__}

    def __call__(self, tensors):
        with self.torch.inference_mode():
            result = self.model(self.torch.from_numpy(np.stack(tensors))).float().numpy()
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if result.shape != (len(tensors), 512) or not np.isfinite(result).all() or (norms < 1e-8).any():
            raise ValueError('invalid SSCD descriptor')
        return (result / norms).astype(np.float32)


def checked_vector(vector, dimension):
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (dimension,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError('invalid vector')
    return np.ascontiguousarray(v / np.linalg.norm(v), dtype=np.float32)


class Catalog:
    """SQLite is authoritative; HNSW is a rebuildable per-process search view.

    Inserts recheck under BEGIN IMMEDIATE so concurrent batches see one another.
    Feature identity cannot change inside a catalog. HNSW approximates candidate
    search; exact byte/pixel detection is indexed and does not depend on ANN recall.
    """
    def __init__(self, path, identity, threads=8):
        import faiss
        self.faiss = faiss
        faiss.omp_set_num_threads(threads)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
            pixel_sha256 TEXT UNIQUE NOT NULL, embedding BLOB NOT NULL, metadata TEXT NOT NULL)''')
        self.db.execute('CREATE TABLE IF NOT EXISTS aliases (sha256 TEXT PRIMARY KEY, canonical_asset_id TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS observations (observation_key TEXT PRIMARY KEY, asset_id TEXT NOT NULL, metadata TEXT NOT NULL)')
        self.db.commit()
        encoded = json.dumps(identity, sort_keys=True)
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('model_identity', encoded))
            existing = self.db.execute('SELECT value FROM metadata WHERE key=?', ('model_identity',)).fetchone()[0]
            if existing != encoded:
                raise ValueError('SSCD model/preprocess identity differs from historical catalog')
        self.dimension = identity['dimensions']
        self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 160
        self.index.hnsw.efSearch = 128
        self.ids = []
        self.last_rowid = 0
        self.sync_index()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            # A failed transaction may have appended uncommitted rows to the
            # in-memory ANN view. It must be rebuilt before the next query.
            if hasattr(self, 'index'):
                self.index.reset()
                self.ids = []
                self.last_rowid = 0
                self.sync_index()
            raise

    def sync_index(self):
        cursor = self.db.execute('SELECT rowid,asset_id,embedding FROM assets WHERE rowid>? ORDER BY rowid', (self.last_rowid,))
        while block := cursor.fetchmany(1024):
            vectors = np.stack([np.frombuffer(row[2], dtype=np.float32) for row in block])
            self.index.add(np.ascontiguousarray(vectors))
            self.ids.extend(row[1] for row in block)
            self.last_rowid = block[-1][0]

    def cached(self, sha):
        found = self.db.execute('SELECT a.asset_id,a.embedding FROM aliases x JOIN assets a ON a.asset_id=x.canonical_asset_id WHERE x.sha256=?', (sha,)).fetchone()
        if found is None:
            found = self.db.execute('SELECT asset_id,embedding FROM assets WHERE sha256=?', (sha,)).fetchone()
        return None if found is None else (found[0], np.frombuffer(found[1], dtype=np.float32).copy())

    def integrate(self, entries, candidate_threshold=.75, top_k=20):
        if not -1 <= candidate_threshold <= 1 or top_k < 1:
            raise ValueError('invalid candidate search parameters')
        results, relations = [], []
        before = self.last_rowid
        with self.transaction():
            self.sync_index()  # Recheck everything committed since this worker's read snapshot.
            for metadata, raw in entries:
                vector = checked_vector(raw, self.dimension)
                sha, pixel, asset_id = metadata['sha256'], metadata['pixel_sha256'], metadata['asset_id']
                identity_row = self.db.execute('SELECT sha256,pixel_sha256 FROM assets WHERE asset_id=?', (asset_id,)).fetchone()
                if identity_row and identity_row != (sha, pixel):
                    raise ValueError('asset_id reused for different content')
                cached = self.cached(sha)
                exact_kind = 'byte' if cached else None
                canonical = cached[0] if cached else None
                if canonical is None:
                    found = self.db.execute('SELECT asset_id FROM assets WHERE pixel_sha256=?', (pixel,)).fetchone()
                    if found:
                        canonical, exact_kind = found[0], 'pixel'
                candidates = []
                if canonical is None:
                    if self.index.ntotal:
                        scores, indices = self.index.search(vector.reshape(1, -1), min(top_k, self.index.ntotal))
                        candidates = [{'other_asset_id': self.ids[int(i)], 'sscd_cosine': float(score),
                                       'decision': 'requires_pair_review', 'automatic_merge': False}
                                      for score, i in zip(scores[0], indices[0]) if i >= 0 and score >= candidate_threshold]
                        relations.extend({'asset_id': asset_id, **candidate} for candidate in candidates)
                    canonical = asset_id
                    self.db.execute('INSERT INTO assets VALUES (?,?,?,?,?)',
                                    (asset_id, sha, pixel, vector.tobytes(), json.dumps(metadata, ensure_ascii=False)))
                    self.sync_index()  # Include earlier members of this same batch.
                self.db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?)', (sha, canonical))
                observation = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                key = hashlib.sha256(observation.encode()).hexdigest()
                self.db.execute('INSERT OR IGNORE INTO observations VALUES (?,?,?)', (key, canonical, observation))
                results.append({'asset_id': asset_id, 'canonical_asset_id': canonical,
                                'sha256': sha, 'pixel_sha256': pixel,
                                'exact_duplicate': exact_kind is not None, 'exact_duplicate_kind': exact_kind,
                                'sscd_candidates': candidates, 'training_eligibility': None,
                                'supervision_records_dropped': 0})
        return {'assets': results, 'relations': relations, 'history_revision_before': before,
                'history_revision_after': self.last_rowid}

    def close(self):
        self.db.close()


def process_assets(assets, *, output_dir, history_path=None, model_path=DEFAULT_MODEL,
                   batch_size=16, threads=8, candidate_threshold=.75, top_k=20):
    """Validate, embed, search and commit an explicitly bounded work unit.

    Call repeatedly with immutable work units, sharing history_path. Source refs
    and supervision stay with callers. A failed work unit has no COMPLETE marker;
    content-addressed cached features make a new attempt idempotent.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if batch_size < 1 or threads < 1:
        raise ValueError('batch_size/threads must be positive')
    start = time.monotonic()
    assets = list(assets)
    if len({a['asset_id'] for a in assets}) != len(assets):
        raise ValueError('asset_id must be unique inside a work unit')
    encoder = Encoder(model_path, threads)
    catalog = Catalog(history_path or output / 'assets.sqlite', encoder.identity, threads)
    rows, relations, failed = [], [], []
    count_encoded, count_cached = 0, 0
    before = catalog.last_rowid
    try:
        with ThreadPoolExecutor(max_workers=min(threads, 16)) as pool:
            for offset in range(0, len(assets), batch_size):
                def checked(asset):
                    try:
                        return load_image_asset(asset), None
                    except Exception as exc:
                        return None, {'asset_id': asset['asset_id'], 'error': f'{type(exc).__name__}: {exc}',
                                      'training_eligibility': None}
                loaded = list(pool.map(checked, assets[offset:offset + batch_size]))
                pending, vectors, metadata = [], {}, []
                for value, error in loaded:
                    if error:
                        failed.append(error)
                        continue
                    meta, tensor = value
                    metadata.append(meta)
                    cached = catalog.cached(meta['sha256'])
                    if cached:
                        vectors[meta['asset_id']] = cached[1]
                        count_cached += 1
                    else:
                        pending.append((meta, tensor))
                if pending:
                    generated = encoder([t for _, t in pending])
                    for (meta, _), vector in zip(pending, generated):
                        vectors[meta['asset_id']] = vector
                    count_encoded += len(pending)
                result = catalog.integrate([(m, vectors[m['asset_id']]) for m in metadata], candidate_threshold, top_k)
                rows.extend(result['assets'])
                relations.extend(result['relations'])
        receipt = {'model_identity': encoder.identity, 'model_path': str(Path(model_path).resolve()),
                   'history_path': str(catalog.path.resolve()), 'history_revision_before': before,
                   'history_revision_after': catalog.last_rowid, 'input_assets': len(assets),
                   'checked_assets': len(rows), 'failed_assets': len(failed), 'features_computed': count_encoded,
                   'features_reused': count_cached, 'exact_duplicates': sum(r['exact_duplicate'] for r in rows),
                   'candidate_edges': len(relations), 'candidate_threshold': candidate_threshold,
                   'candidate_threshold_is_deletion_threshold': False,
                   'search': {'algorithm': 'faiss_hnsw_inner_product', 'M': 32, 'efConstruction': 160,
                              'efSearch': 128, 'top_k': top_k, 'approximate_candidates': True},
                   'minimum_original_short_edge': 128, 'automatic_near_duplicate_merge': False,
                   'duration_seconds': time.monotonic() - start, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                   'code_sha256': file_sha(__file__), 'published_training': False}
        result = {'assets': rows, 'relations': relations, 'failed': failed, 'receipt': receipt}
        atomic_json(output / 'result.json', result)
        atomic_json(output / 'COMPLETE.json', {'asset_stage_completed': True,
                    'all_assets_passed': not failed, 'result_sha256': file_sha(output / 'result.json'),
                    'published_training': False})
        return result
    finally:
        catalog.close()
