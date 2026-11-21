"""Replay published examples with the production reader and Wan packer on the mounted originals."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from configs.config import RunConfig
from corpus_pipeline.production.common import file_sha
from data.packing import Packer, token_contract
from data.sources import release_geometry
from data.stream import DocumentStream


class ExampleStream(DocumentStream):
    """Limit each pack to the selected catalog coordinate."""
    def _fill(self):
        pass

    def peek(self):
        return self.rows[0] if self.rows and self.row_index == 0 else None

    def advance(self):
        self.row_index += 1

    def example(self, root, case, view):
        indices = []
        parent = root
        for name in case['leaf'].split('/'):
            sources = json.loads((parent / 'manifest.json').read_bytes())['sources']
            indices.append([s['path'] for s in sources].index(name))
            parent = parent / name
        coordinate = case['file']
        for index in reversed(indices):
            coordinate = (index, coordinate)
        self.signature.update(view=view, epoch=case['epoch'])
        self.rows = self._load(coordinate, case['selected_row'], 1)['rows']
        self.row_index = 0
        return self.rows[0]


def verify(root):
    root = Path(root).resolve()
    package = root.parent
    published = json.loads((package / 'PACKAGE.json').read_bytes())
    assert file_sha(root / 'manifest.json') == published['manifest_sha256']
    assert file_sha(root / 'source_roots.json') == published['source_roots_sha256']
    for mount in published['mounts']:
        actual, options = subprocess.check_output(
            ['findmnt', '-n', '-T', mount, '-o', 'TARGET,OPTIONS'], text=True
        ).split()
        assert actual == mount and 'ro' in options.split(',')
    cases = json.loads((package / 'READ_CASES.json').read_bytes())
    cfg = RunConfig.load(Path(__file__).resolve().parents[1] / 'configs/qwen35_2b_wan.yaml')
    cfg.vision_assets = cfg.text_assets = str(package / 'processor')
    length = release_geometry(root)['sequence_length']
    torch.set_num_threads(1)
    stream = ExampleStream(root, 'A', 0, 1, 42, cfg=cfg,
                           workers=1, prefetch=1, image_threads=1, image_prefetch=1, sequence_length=length)
    try:
        assert stream.catalog.meta['documents'] == published['records']
        packer = Packer(stream, cfg, length, token_contract(root, cfg), padding_multiple=128)
        checks = []
        for case in cases:
            result = dict(case=case['case'], leaf=case['leaf'])
            for view in ('A', 'B'):
                doc = stream.example(root, case, view)
                images = [e for e in doc['elements'] if e['type'] == 'image']
                assert doc['id'] == case['sample_id']
                assert [i['locator']['sha256'] for i in images] == (case['sha'] if view == 'A' else [])
                targets = [i+1 for i,e in enumerate(images) if e['image_loss_allowed'] and e['generation_target']]
                batch, receipt = packer.next()
                assert len(receipt['documents']) == 1 and receipt['documents'][0]['completed']
                assert receipt['occupied'] <= length and receipt['images'] == len(images)
                positions = [i+1 for i in batch.generation_image_indices.tolist()]
                assert positions == targets
                stats = dict(occupied=receipt['occupied'], ce=len(batch.text_indices), flow=len(positions),
                             images=len(images), flow_tokens=receipt['generation_tokens'])
                stats['target_image_positions'] = positions
                assert stats == case['expected'][view], (case['case'], view, stats)
                result[view] = stats
            checks.append(result)
        def leaves(catalog):
            if hasattr(catalog, 'catalogs'):
                for child in catalog.catalogs:
                    yield from leaves(child)
            else:
                yield str(catalog.path.relative_to(root))

        assert {c['leaf'] for c in checks} == set(leaves(stream.catalog))
        return dict(passed=True, records=published['records'], manifest_sha256=published['manifest_sha256'],
                    mounts=published['mounts'], cases=checks)
    finally:
        stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release')
    args = parser.parse_args()
    print(json.dumps(verify(args.release), ensure_ascii=False))
