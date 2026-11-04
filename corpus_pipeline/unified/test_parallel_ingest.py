import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from PIL import Image

from corpus_pipeline.production.common import atomic_json, file_sha
from corpus_pipeline.production.records import snapshot
from .genund_loose import BLIP, prepare_blip_units
from .parallel_ingest import worker, registered, finalize_source


def setup(tmp_path):
    root=tmp_path/'mount'; raw=root/BLIP/'raw'; raw.mkdir(parents=True)
    Image.new('RGB',(160,128),'red').save(raw/'image.jpg')
    (raw/'image.txt').write_text('Original supervision')
    run=tmp_path/'run'; (run/'ingest_registry').mkdir(parents=True)
    units=prepare_blip_units(root,run/'plan',dataset_id='inspire/genund@'+'a'*64)
    path=run/'plan/PLAN.json'
    reg={'complete':True,'source':BLIP,'plan':str(path),'plan_sha256':file_sha(path)}
    atomic_json(run/'ingest_registry/blip.json',reg)
    return root,run,reg,units


def test_atomic_source_snapshot_and_views(tmp_path):
    root,run,reg,units=setup(tmp_path)
    assert finalize_source(run,reg,units) is None
    worker(run,'fixture',once=True)
    result=finalize_source(run,reg,units)
    assert result['statistics']['samples']==2
    out=run/'normalized_sources'/BLIP
    snap=snapshot(out)
    rows=[row for path in snap['catalog_index'] for row in pq.read_table(path).to_pylist()]
    original=[json.loads(row['original_supervision_json']) for row in rows]
    assert [row['image_loss_allowed'] for row in original]==[False,True]
    assert rows[0]['images']==rows[1]['images']
    checksum=file_sha(out/'COMPLETE.json')
    worker(run,'fixture',once=True)
    assert file_sha(out/'COMPLETE.json')==checksum
    assert not (run/'COMPLETE.json').exists()  # Five undeclared sources remain pending.


def test_plan_mutation_is_rejected(tmp_path):
    _,run,reg,_=setup(tmp_path)
    Path(reg['plan']).write_text('{}')
    with pytest.raises(ValueError,match='plan changed'): registered(run)


def test_failure_never_publishes_source(tmp_path):
    root,run,reg,units=setup(tmp_path)
    (root/BLIP/'raw/image.txt').unlink()
    worker(run,'fixture',once=True)
    assert finalize_source(run,reg,units) is None
    errors=list((run/'normalized_sources'/BLIP/'claims').glob('*.error.json'))
    assert len(errors)==1
    assert json.loads(errors[0].read_text())['attempts']==1
