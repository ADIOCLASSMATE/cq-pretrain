"""Read original native records selected by an auditable candidate index.

No recaption, task reversal, representative-image substitution or image decoding
occurs here. ``keep`` is the deduplication decision, not an AI/quality warranty.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import digest, file_sha, require, stamp
from .training_release import iter_candidate_references, text_fingerprint


class CandidateReader:
    def __init__(self, output, max_open_files=16):
        self.output=Path(output).resolve(strict=True)
        complete=json.loads((self.output/'COMPLETE.json').read_text())
        require(complete.get('complete') is True,'Candidate index incomplete')
        manifest_path=self.output/'manifest.json'
        require(file_sha(manifest_path)==complete['manifest_sha256'],'Candidate manifest changed')
        manifest=json.loads(manifest_path.read_text())
        self.native_bindings={Path(x['path']).name:x for x in manifest['native_metadata_bindings']}
        self.max_open_files=max_open_files
        self.cache=OrderedDict()

    def close(self):
        for reader,_ in self.cache.values():reader.close()
        self.cache.clear()

    def _source(self,path):
        path=str(Path(path).resolve(strict=True))
        if path in self.cache:
            reader,binding=self.cache.pop(path)
            require(stamp(path)==binding['stamp'],'Native metadata changed after opening')
            self.cache[path]=(reader,binding)
            return reader
        name=hashlib.sha256(path.encode()).hexdigest()+'.json'
        require(name in self.native_bindings,'Native file absent from published index bindings')
        artifact=self.native_bindings[name]; binding_path=self.output/artifact['path']
        require(file_sha(binding_path)==artifact['sha256'],'Native binding receipt changed')
        binding=json.loads(binding_path.read_text())
        require(binding['path']==path and stamp(path)==binding['stamp'],'Native file identity changed')
        require(file_sha(path)==binding['sha256'],'Native metadata bytes changed')
        require(stamp(path)==binding['stamp'],'Native metadata changed during hashing')
        reader=pq.ParquetFile(path); self.cache[path]=(reader,binding)
        while len(self.cache)>self.max_open_files:
            _,(old,_)=self.cache.popitem(last=False);old.close()
        return reader

    def read(self,reference,*,allow_excluded=False):
        require(allow_excluded or reference['decision']=='keep','Reference is not in keep view')
        source=self._source(reference['original_parquet_path'])
        target=reference['row_index']
        require(type(target) is int and 0<=target<source.metadata.num_rows,'Invalid native row locator')
        offset=0;record=None
        for group in range(source.num_row_groups):
            count=source.metadata.row_group(group).num_rows
            if target>=offset+count:offset+=count;continue
            for batch in source.iter_batches(row_groups=[group],batch_size=128):
                if target<offset+batch.num_rows:
                    record=batch.slice(target-offset,1).to_pylist()[0];break
                offset+=batch.num_rows
            break
        require(record is not None and record['sample_id']==reference['sample_id'],'Native sample identity differs')
        require(digest(record)==reference['source_record_sha256'],'Native full record SHA differs from frontpass')
        require(text_fingerprint(record['segments'])==record['text_hash']==reference['text_hash'],
                'Supervision text/order differs')
        require(record['format']==reference['native_format'],'Native task format differs')
        require([x['sha256'] for x in record['images']]==reference['image_ids'],'Image identity/order differs')
        return record

    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def iter_selected_records(output,limit=None):
    """Yield (original native record, selection reference) without decoding images."""
    with CandidateReader(output) as reader:
        for n,reference in enumerate(iter_candidate_references(output,decision='keep')):
            if limit is not None and n>=limit:break
            yield reader.read(reference),reference
