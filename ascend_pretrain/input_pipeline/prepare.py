"""Generate explicitly synthetic local DCVLM fixtures; never download images."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tokenizers import Tokenizer

from .images import atomic_json


def make_fake(output, tokenizer_root, rows=131072, shard_rows=2048, edge_cases=False):
    output=Path(output).resolve(); tokenizer_root=Path(tokenizer_root)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'data').mkdir();(output/'raw').mkdir();(output/'tokenizer').mkdir()
    lock_files=[]
    for name in ('tokenizer.json','tokenizer_config.json'):
        src=tokenizer_root/name;dst=output/'tokenizer'/name
        shutil.copyfile(src,dst)
        lock_files.append(dict(file=name,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size))
    tokenizer=Tokenizer.from_file(str(output/'tokenizer/tokenizer.json'))
    excluded=set(range(151643,151657))
    rng=np.random.default_rng(42);images=[]
    with tarfile.open(output/'raw/fake.tar','w',format=tarfile.USTAR_FORMAT) as archive:
        for i,(width,height) in enumerate([(256,256),(384,256),(256,384),(512,512),(1024,512),(512,768)]):
            buffer=io.BytesIO()
            Image.fromarray(rng.integers(0,256,(height,width,3),dtype=np.uint8)).save(buffer,format='JPEG',quality=85)
            body=buffer.getvalue();member=tarfile.TarInfo(f'{i}.jpg');member.size=len(body)
            offset=archive.offset+512;archive.addfile(member,io.BytesIO(body))
            images.append(dict(member=member.name,offset=offset,length=len(body),sha256=hashlib.sha256(body).hexdigest(),
                               width=width,height=height,format='JPEG'))
    # Raw UTF-8 text is always stored. Fixture construction computes only the
    # validation count; training must run its real online tokenizer again.
    texts=['A photograph of a green tree beside a quiet river. 图像中的树和河流。',
           'Explain this example step by step. Numbers 123 and words café 自然语言。 '*32,
           'The following illustrated document describes a scene. '*16,
           'Second paragraph with an image and a detailed description. '*20,
           '', '<|im_start|>This literal special token stays in the encoded span.']
    if edge_cases:texts.append('Long document fragment, 中文保持顺序. '*3000)
    counts={x:sum(t not in excluded for t in tokenizer.encode(x,add_special_tokens=False).ids) for x in texts}
    catalog=[];statistics=dict(samples=rows,text_tokens=0,image_occurrences=0)
    schema=pa.schema([('sample_id',pa.string()),('source_shard',pa.string()),
        ('segments',pa.list_(pa.struct([('kind',pa.string()),('text',pa.string()),('image_index',pa.int32())]))),
        ('images',pa.list_(pa.struct([('member',pa.string()),('offset',pa.int64()),('length',pa.int64()),
            ('sha256',pa.string()),('width',pa.int32()),('height',pa.int32()),('format',pa.string())]))),
        ('text_tokens',pa.int64()),('format',pa.string())])
    def text(value):return dict(kind='text',text=value,image_index=None)
    def image(index):return dict(kind='image',text=None,image_index=index)
    for start in range(0,rows,shard_rows):
        records=[]
        for index in range(start,min(rows,start+shard_rows)):
            mode=index%16;chosen=[]
            if mode<12:
                segments=[image(0),text(texts[0])];chosen=[images[index%len(images)]];kind='caption'
            elif mode<15:
                segments=[text(texts[1])];kind='text_conversation'
            else:
                chosen=[images[(index+j)%len(images)] for j in range(4)]
                segments=[text(texts[2]),image(0),text(texts[3]),image(1),image(2),text(texts[4]),image(3)]
                kind='interleaved'
            if edge_cases and index%17==0:segments.insert(0,text(texts[-1]))
            if edge_cases and index%13==0:segments.append(text(texts[5]))
            ntext=sum(counts[s['text']] for s in segments if s['kind']=='text')
            records.append(dict(sample_id=f'fake-{index:012d}',source_shard='fake.tar',segments=segments,
                                images=chosen,text_tokens=ntext,format=kind))
            statistics['text_tokens']+=ntext;statistics['image_occurrences']+=len(chosen)
        path=f'data/{start//shard_rows:05d}.parquet'
        pq.write_table(pa.Table.from_pylist(records,schema=schema),output/path,row_group_size=min(shard_rows,256),compression='zstd')
        catalog.append(dict(path=path,rows=len(records),bytes=(output/path).stat().st_size,source='synthetic',source_shard='fake.tar'))
    atomic_json(output/'catalog.json',catalog)
    atomic_json(output/'sources.json',dict(raw_shards=[dict(path='fake.tar',bytes=(output/'raw/fake.tar').stat().st_size)]))
    atomic_json(output/'manifest.json',dict(schema_version='dcvlm.pretrain.v1',phase='complete',synthetic=True,
        source_pool=str(output/'raw'),statistics=statistics,
        tokenizer_lock=dict(files=lock_files,counting=dict(exclude_special_token_ids=sorted(excluded))),
        image_policy=dict(counted_tokens_per_image=512),
        fixture_note='Controlled synthetic mixture; repeated local JPEGs. Not a cold-I/O or corpus-time estimate.'))
    atomic_json(output/'reports/validation.json',dict(passed=True,synthetic=True))
    atomic_json(output/'COMPLETE.json',dict(validation_passed=True,
        catalog_sha256=hashlib.sha256((output/'catalog.json').read_bytes()).hexdigest()))
    return statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--rows',type=int,default=131072);parser.add_argument('--shard-rows',type=int,default=2048)
    parser.add_argument('--edge-cases',action='store_true')
    args=parser.parse_args()
    print(json.dumps(make_fake(args.output,args.tokenizer,args.rows,args.shard_rows,args.edge_cases)))


if __name__=='__main__':main()
