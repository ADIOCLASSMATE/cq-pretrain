"""Recover exact missing/empty ScaleCap images from pinned official ZIP bytes."""
import base64
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import struct
import sys
import tempfile
import zipfile
import zlib

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from corpus_pipeline.production.common import atomic_json,digest,file_sha

RUN=Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1')
ROOT=Path('/inspire/dataset/gen-und-mm-pretrain/01'); out=RUN/'scalecap_asset_recovery'; out.mkdir(exist_ok=True)
info=json.loads((RUN/'annotations/ScaleCap-450k/hub_info.json').read_text())
audit=json.loads((RUN/'scalecap_full_path_audit.json').read_text()); targets=set(audit['missing_distinct_paths'])
for leaf in ['share4v_image','laion_filter_image']:
    for entry in os.scandir(ROOT/'ScaleCap450k_image/images'/leaf):
        if entry.is_file() and entry.stat().st_size==0:
            targets.add('ScaleCap450k_image/images/'+leaf+'/'+entry.name)
archives=[item for item in info['siblings'] if item['rfilename'].endswith('.zip')]
identity={'repo':'long-xing1/ScaleCap-450k','revision':info['sha'],'annotation_sha256':next(item['lfs']['sha256'] for item in info['siblings'] if item['rfilename']=='ScaleCap450k.json'),
          'zip_inventory_sha256':digest(archives),'contract':'corpus.scalecap.exact_zip_recovery.v1'}
binding={**identity,'dataset_id':'inspire/gen-und-scalecap-recovered@'+digest(identity),
         'root':str(RUN/'recovered_assets/ScaleCap'),'source_modified':False}
atomic_json(out/'DATASET_BINDING.json',binding); recovered={}; candidates={name:[] for name in targets}; zero=[]
for archive in archives:
    group=PurePosixPath(archive['rfilename']).parts[1]
    path=ROOT/'ScaleCap450k_image/images'/(group+'_zips')/PurePosixPath(archive['rfilename']).name
    if not path.is_file(): raise ValueError('declared mounted ZIP missing: '+str(path))
    with zipfile.ZipFile(path) as stream:
        for member in stream.infolist():
            if member.is_dir(): continue
            raw=member.filename.encode('utf8' if member.flag_bits&0x800 else 'cp437')
            variants=[('standard_zip_filename',member.filename)]
            if not member.flag_bits&0x800: variants.append(('exact_latin1_annotation_to_raw_zip_bytes',raw.decode('latin1')))
            cursor=0
            while cursor+4<=len(member.extra):
                tag,size=struct.unpack_from('<HH',member.extra,cursor); value=member.extra[cursor+4:cursor+4+size]; cursor+=4+size
                if tag==0x7075 and len(value)>=5 and value[0]==1 and struct.unpack_from('<I',value,1)[0]==zlib.crc32(raw):
                    variants.append(('zip_unicode_path_extra',value[5:].decode('utf8')))
            matched={}
            for basis,name in variants:
                parts=PurePosixPath(name).parts
                if '..' in parts or PurePosixPath(name).is_absolute(): continue
                if len(parts)>1 and tuple(parts[:-1]) not in {(group,),('images',group)}: continue
                target='ScaleCap450k_image/images/'+group+'/'+parts[-1]
                if target in targets: matched[target]=basis
            for target,basis in matched.items():
                candidates[target].append({'archive':archive,'mounted_archive':str(path),'member':member.filename,
                    'member_raw_name_b64':base64.b64encode(raw).decode(),'member_size':member.file_size,
                    'member_crc32':member.CRC,'filename_binding':basis})
verified_archives={}; unresolved=[]
for target,matches in sorted(candidates.items()):
    if len(matches)!=1:
        unresolved.append({'annotation_path':target,'reason':'missing_or_ambiguous_original_zip_member','candidates':matches}); continue
    match=matches[0]; archive=match['archive']; path=Path(match['mounted_archive'])
    if str(path) not in verified_archives:
        sha=file_sha(path)
        if path.stat().st_size!=archive['size'] or sha!=archive['lfs']['sha256']: raise ValueError('mounted ZIP differs from pinned official LFS: '+str(path))
        verified_archives[str(path)]=sha
        atomic_json(out/'VERIFIED_ARCHIVES.json',verified_archives)
    if match['member_size']==0:
        unresolved.append({'annotation_path':target,'reason':'zero_byte_member_in_official_lfs_verified_zip','evidence':match}); continue
    with zipfile.ZipFile(path) as stream: payload=stream.read(match['member'])
    sha=hashlib.sha256(payload).hexdigest(); rel=Path('ScaleCap')/sha[:2]/(sha+'.img'); dest=Path(binding['root'])/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        descriptor,name=tempfile.mkstemp(prefix='.'+sha+'.',suffix='.partial',dir=dest.parent)
        with os.fdopen(descriptor,'wb') as handle: handle.write(payload)
        os.replace(name,dest)
    if file_sha(dest)!=sha: raise ValueError('recovered image SHA differs')
    recovered[target]={'path':str(rel),'dataset_id':binding['dataset_id'],'sha256':sha,'size':len(payload),
                       'original_annotation_path':target,'source_zip_evidence':match,
                       'official_zip_lfs_sha_verified':True}
    atomic_json(out/'RECOVERY_PROGRESS.json',{'recovered':len(recovered),'targets':len(targets),'unresolved':len(unresolved)})
result={'complete':True,'binding':binding,'images':recovered,'unresolved':unresolved,'target_count':len(targets),
        'recovered_count':len(recovered),'verified_archives':verified_archives,'source_files_modified':False,
        'image_payloads_persisted':bool(recovered),'image_payload_copy_scope':'only_missing_or_empty_exact_original_zip_images'}
atomic_json(out/'IMAGE_RECOVERY_MAP.json',result)
print(json.dumps({'targets':len(targets),'recovered':len(recovered),'unresolved':len(unresolved),'binding':binding}))
