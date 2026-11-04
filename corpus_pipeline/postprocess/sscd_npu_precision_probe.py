"""Backend numerical qualification evidence; not full-corpus recall certification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu


def main():
    p=argparse.ArgumentParser();p.add_argument('--vectors',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False
    options={'ACL_OP_SELECT_IMPL_MODE':'high_precision','ACL_OPTYPELIST_FOR_IMPLMODE':'MatMul'}
    torch.npu.set_option(options)
    assert torch.npu.matmul.allow_hf32 is False
    rng=np.random.default_rng(20125)
    def normalize(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
    x=normalize(rng.normal(size=(1024,512)))
    z=rng.normal(size=x.shape);z=normalize(z-(x*z).sum(1,keepdims=True)*x)
    eps=np.tile(np.array([-1e-5,-1e-6,-1e-7,0,1e-7,1e-6,1e-5,2e-5]),128)
    alpha=.75+eps;y=alpha[:,None]*x+np.sqrt(1-alpha**2)[:,None]*z
    cancelling=normalize(np.tile(np.where(np.arange(512)%2,1.,-1.),(1024,1))+rng.normal(0,1e-4,(1024,512)))
    raw=np.memmap(a.vectors,dtype=np.float32,mode='r').reshape(-1,512)
    families=[('real',np.array(raw[:1024]),np.array(raw[1024:3072])),
              ('near_threshold',x.astype(np.float32),y.astype(np.float32)),
              ('cancellation',cancelling.astype(np.float32),normalize(rng.normal(size=(1024,512))).astype(np.float32))]
    results=[]
    for name,q,d in families:
        ref64=q.astype(np.float64)@d.astype(np.float64).T;ref32=q@d.T
        qn=torch.from_numpy(q).to('npu');dn=torch.from_numpy(d).to('npu')
        for dtype in (torch.float32,torch.float16):
            result=(qn.to(dtype)@dn.to(dtype).T).float().cpu().numpy();err=np.abs(result-ref64)
            row=dict(family=name,dtype=str(dtype),pairs=int(err.size),max_abs_error_vs_float64=float(err.max()),
                     max_abs_error_vs_cpu_fp32=float(np.abs(result-ref32).max()),
                     p999_error=float(np.quantile(err,.999)),positives=int((ref64>=.75).sum()),
                     missed_positives_margin_2e6=int(((ref64>=.75)&(result<.75-2e-6)).sum()),
                     missed_positives_margin_01=int(((ref64>=.75)&(result<.74)).sum()))
            if dtype==torch.float16:
                roundref=(q.astype(np.float16).astype(np.float64)@d.astype(np.float16).astype(np.float64).T).astype(np.float16).astype(np.float32)
                row['max_error_vs_fp64_accum_half_inputs_half_output']=float(np.abs(result-roundref).max())
            results.append(row);print(json.dumps(row),flush=True)
    cann={}
    for filename in ('/usr/local/Ascend/ascend-toolkit/latest/version.cfg',
                     '/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info',
                     '/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info'):
        path=Path(filename)
        if path.exists():cann[filename]=path.read_text()[:2000]
    worst=max(x['max_abs_error_vs_float64'] for x in results if x['dtype']=='torch.float32')
    value={'schema':'corpus.sscd.npu.backend_numerical_qualification.v1','created_at':time.time(),
           'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device':torch.npu.get_device_name(0),
           'allow_hf32':torch.npu.matmul.allow_hf32,'jit_compile':False,'requested_options':options,
           'options_readback':{key:str(torch_npu._C._npu_getOption(key)) for key in options},'cann_files':cann,
           'dtype':'float32','worst_observed_error':worst,'tested_margin':2e-6,
           'qualified':worst<2e-6,'qualification_scope':'sampled_backend_numeric_error_only',
           'full_catalog_recall_qualified':False,'fp16_accumulation_proven':False,
           'results':results,'source_vectors':a.vectors,'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(value),flush=True)


if __name__=='__main__':main()
