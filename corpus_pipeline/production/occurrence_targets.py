"""Materialize a native catalog's uniform row grants on each visual occurrence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data.loss_permissions import TARGET_FIELDS


def write_shard(source, destination, *, direction=None):
    parquet = pq.ParquetFile(source)
    table = parquet.read().combine_chunks()
    grants = table['image_loss_allowed'].chunk(0)
    segments = table['segments'].chunk(0)
    spans = segments.values
    asset_fields = table['images'].type.value_type.names
    nested = set(asset_fields) | set(spans.type.names)
    if (grants.type != pa.bool_() or grants.null_count or segments.null_count or spans.null_count or spans.field('kind').null_count
            or nested.intersection((*TARGET_FIELDS, 'upstream_image', 'metadata_json'))
            or set(table.column_names).intersection(('dcvlm_image_loss_allowed','generation_target'))):
        raise ValueError('Expected a native catalog with a single explicit boolean grant per row')
    images = pc.equal(spans.field('kind'), 'image')
    targets = pc.if_else(images, pc.take(grants, pc.list_parent_indices(segments)), pa.scalar(None, pa.bool_()))
    extended = pa.StructArray.from_arrays([*(spans.field(k) for k in spans.type.names), targets],
                                          names=[*spans.type.names, 'generation_target'])
    output = table.set_column(table.schema.get_field_index('segments'), 'segments',
                              pa.ListArray.from_arrays(segments.offsets, extended))
    if direction is not None:
        if direction not in ('I2T', 'T2I') or 'direction' in table.column_names:
            raise ValueError('Expected one new I2T or T2I direction for the catalog')
        output = output.append_column('direction', pa.repeat(pa.scalar(direction), len(output)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, destination, compression='zstd', row_group_size=parquet.metadata.row_group(0).num_rows)
    if not pq.read_table(destination).equals(output):
        raise ValueError(f'Published Parquet differs from the verified projection: {destination}')
    return len(table), int(pc.sum(images).as_py() or 0), int(pc.sum(targets).as_py() or 0)


def publish(source, destination, *, direction=None):
    source, destination = Path(source).resolve(strict=True), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    catalog_bytes = (source/'catalog.json').read_bytes()
    done = json.loads((source/'COMPLETE.json').read_bytes())
    manifest = json.loads((source/'manifest.json').read_bytes())
    validation = json.loads((source/'reports/validation.json').read_bytes())
    if (manifest.get('schema_version') != 'dcvlm.pretrain.v1' or manifest.get('phase') != 'complete'
            or not validation['passed'] or not done['validation_passed']
            or hashlib.sha256(catalog_bytes).hexdigest() != done['catalog_sha256']):
        raise ValueError('Source catalog is incomplete or changed')
    def dump(path, value):
        path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp)/'catalog'
        stage.mkdir()
        output = []
        occurrences = targets = records = 0
        for shard in json.loads(catalog_bytes):
            path = Path(shard['path'])
            if path.is_absolute() or '..' in path.parts or (source/path).stat().st_size != shard['bytes']:
                raise ValueError('Invalid native shard path or size')
            n, images, enabled = write_shard(source/path, stage/path, direction=direction)
            if n != shard['rows']:
                raise ValueError('Native shard row count differs from the published catalog')
            published = dict(shard, bytes=(stage/path).stat().st_size)
            if 'sha256' in shard:
                if hashlib.sha256((source/path).read_bytes()).hexdigest() != shard['sha256']:
                    raise ValueError('Native shard SHA-256 differs from the published catalog')
                published['sha256'] = hashlib.sha256((stage/path).read_bytes()).hexdigest()
            output.append(published)
            records += n
            occurrences += images
            targets += enabled
        for name in ('manifest.json','sources.json'):
            shutil.copy2(source/name, stage/name)
        (stage/'tokenizer').symlink_to((source/'tokenizer').resolve(strict=True))
        (stage/'reports').mkdir()
        audit = dict(validation,records=records,
                     image_occurrences=occurrences,generation_targets=targets,all_original_columns_verified=True)
        dump(stage/'reports/validation.json', audit)
        dump(stage/'catalog.json', output)
        dump(stage/'COMPLETE.json', dict(done,catalog_sha256=hashlib.sha256((stage/'catalog.json').read_bytes()).hexdigest()))
        stage.rename(destination)
        return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--direction', choices=('I2T', 'T2I'), help='Declare one task direction for all published rows')
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination, direction=args.direction)))
