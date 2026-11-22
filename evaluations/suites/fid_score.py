"""Extract ADM pool_3 features on CPU while sampling runs, then score FID-50K.

Runs with the container's Python and the TensorFlow environment under
<benchmarks>/fid/tf-env; it imports nothing from this repository.
Usage: python3.12 -S fid_score.py OUTPUT BENCHMARKS
"""
import json,os,sys,time
from pathlib import Path
import hashlib
import site

# The job uses the container's Python and the shared environment's packages.
# Process its .pth files as a normal venv startup would.
FID=Path(sys.argv[2])/'fid'
site.addsitedir(str(FID/'tf-env/lib/python3.12/site-packages'))
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

sys.path.insert(0,str(FID))
import adm_evaluator as adm
adm.INCEPTION_V3_PATH=str(FID/'classify_image_graph_def.pb')
tf.disable_v2_behavior()
N=50000
CHUNK=32
def atomic_json(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def score_one(out,session,input_batch,activations):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').is_file():
        return
    feat=out/'features';feat.mkdir(exist_ok=True)
    pending=set(range((N+CHUNK-1)//CHUNK))
    while pending:
        ready=[]
        for mark in (out/'ready').glob('*.json'):
            idx=int(mark.stem)
            if idx in pending:ready.append(idx)
        if not ready:
            time.sleep(8);continue
        for k in sorted(ready):
            start,end=k*CHUNK,min(N,(k+1)*CHUNK);name=f'{k:05d}'
            final=feat/f'{name}.npy';folder=out/'scratch'/name
            if not final.is_file():
                images=[]
                for i in range(start,end):
                    with Image.open(folder/f'{i:06d}.png') as im:
                        assert im.size==(512,512) and im.mode=='RGB'
                        images.append(np.ascontiguousarray(im))
                values=session.run(activations,feed_dict={input_batch:np.stack(images).astype(np.float32)}).reshape(end-start,2048)
                if not np.isfinite(values).all():raise ValueError('Non-finite Inception features')
                temp=final.with_suffix('.partial')
                with temp.open('wb') as w:np.save(w,values,allow_pickle=False)
                temp.replace(final)
            data=np.load(final,mmap_mode='r')
            assert data.shape==(end-start,2048)
            for i in range(start,end):
                f=folder/f'{i:06d}.png'
                if i<1000 and f.is_file():
                    demo=out/'samples';demo.mkdir(exist_ok=True)
                    f.replace(demo/f.name)
                elif f.is_file():f.unlink()
            if folder.is_dir():folder.rmdir()
            pending.remove(k)
    while not (out/'GENERATED.json').is_file():time.sleep(2)
    ref=json.loads((FID/'reference.json').read_text())
    values=np.concatenate([np.load(x,mmap_mode='r') for x in sorted(feat.glob('*.npy'))])
    assert values.shape==(50000,2048) and np.isfinite(values).all()
    stats=adm.FIDStatistics(values.mean(axis=0),np.cov(values,rowvar=False))
    real=adm.FIDStatistics(np.load(FID/'mu.npy'),np.load(FID/'sigma.npy'))
    fid=float(stats.frechet_distance(real))
    assert np.isfinite(fid) and fid>=0
    np.savez(out/'generated_moments.npz',mu=stats.mu,sigma=stats.sigma)
    generation=json.loads((out/'GENERATED.json').read_text())
    atomic_json(out/'RESULTS.json',dict(**generation,FID=fid,reference=ref,extractor='OpenAI guided-diffusion TensorFlow graph, pool_3, 2048',
                inception_graph_sha256=hashlib.sha256(Path(adm.INCEPTION_V3_PATH).read_bytes()).hexdigest()))
    print(generation['step'],'ImageNet FID-50K',fid,flush=True)
def main():
    config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=12,inter_op_parallelism_threads=2)
    with tf.Session(config=config) as sess:
        image_input=tf.placeholder(tf.float32,shape=[None,None,None,3])
        pool,_=adm._create_feature_graph(image_input)
        score_one(Path(sys.argv[1]),sess,image_input,pool)
if __name__=='__main__':main()
