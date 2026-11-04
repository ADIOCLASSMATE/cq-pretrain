"""Freeze baseline image identities before any in-place repair."""
import json
import os
from pathlib import Path

import duckdb

PUBLIC = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public')
RUN = PUBLIC / 'datasets/_repairs/dcvlm_datacomp_unblur_20261104'
NATIVE = PUBLIC / 'preprocessing/dcvlm_baseline_full_20261028_v3/native'
OLD = PUBLIC / 'datasets/dcvlm_pretrain_v1_20261016T062238Z'
ROOT = PUBLIC / 'datasets/mlfoundations/dcvlm-baseline-200b'


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(config={'threads': 8, 'memory_limit': '12GB'})
    catalog = json.loads((NATIVE / 'catalog.json').read_text())
    save(RUN / 'original_catalog.json', catalog)
    target = RUN / 'targets.parquet'
    if not target.exists():
        paths = [str(NATIVE / r['path']) for r in catalog]
        db.execute(f'''COPY (
            SELECT source_shard, source_key, sample_id,
                   images[1].member AS member, images[1].offset AS offset,
                   images[1].length AS length, images[1].sha256 AS sha256,
                   regexp_extract(json_extract_string(original_supervision_json,
                        '$.orig_url'), '([^/]+[.]tar)$', 1) AS upstream_shard
            FROM read_parquet(?) WHERE source = 'datacomp_1b'
        ) TO '{target}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)''', [paths])
        os.replace(str(target) + '.tmp', target)
    count = db.execute('SELECT count(*), count(DISTINCT sha256), count(DISTINCT upstream_shard) FROM read_parquet(?)', [str(target)]).fetchone()
    assert count[0] == 4968838, count
    assert not db.execute("SELECT count(*) FROM read_parquet(?) WHERE upstream_shard = '' OR sha256 IS NULL", [str(target)]).fetchone()[0]
    print('BASELINE', count, flush=True)
    old_paths = [str(OLD / r['path']) for r in json.loads((OLD / 'catalog.json').read_text()) if r['source'] == 'datacomp_1b']
    matches = RUN / 'old_matches.parquet'
    if not matches.exists():
        db.execute(f'''COPY (
            SELECT DISTINCT w.upstream_shard, w.sha256, o.source_key AS original_key,
                o.images[1].offset AS image_offset, o.images[1].length AS image_length,
                o.images[1].sscd_offset AS sscd_offset,
                o.images[1].source_url AS image_url
            FROM read_parquet(?) o JOIN read_parquet(?) w
            ON o.images[1].sha256 = w.sha256
            AND regexp_extract(o.source_shard, '([^/]+[.]tar)$', 1) = w.upstream_shard
        ) TO '{matches}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)''', [old_paths, str(target)])
        os.replace(str(matches) + '.tmp', matches)
    save(RUN / 'PREPARED.json', dict(records=count[0], unique_images=count[1], upstream_shards=count[2],
        old_index_matches=db.execute('SELECT count(*) FROM read_parquet(?)', [str(matches)]).fetchone()[0],
        root=str(ROOT), native=str(NATIVE)))
    print((RUN / 'PREPARED.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
