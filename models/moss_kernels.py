"""Fused MOSS pointwise forward/backward on the pinned Triton Ascend stack.

Arithmetic uses FP32 intermediates and rounds once to the activation dtype.
The public wrappers in moss.py retain eager execution as an explicit control.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _dense_forward(A,B,C,Y,N,KIND:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    a=tl.load(A+i,mask,0).to(tl.float32)
    b=tl.load(B+i,mask,0).to(tl.float32)
    c=tl.load(C+i,mask,0).to(tl.float32)
    if KIND==1:y=a*(1+b)+c
    else:y=a+b*c
    tl.store(Y+i,y,mask)


@triton.jit(do_not_specialize=['N'])
def _dense_backward(A,B,C,G,DA,DB,DC,N,KIND:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32)
    b=tl.load(B+i,mask,0).to(tl.float32)
    if KIND==1:
        a=tl.load(A+i,mask,0).to(tl.float32)
        tl.store(DA+i,g*(1+b),mask);tl.store(DB+i,g*a,mask)
    else:
        c=tl.load(C+i,mask,0).to(tl.float32)
        tl.store(DB+i,g*c,mask);tl.store(DC+i,g*b,mask)


@triton.jit(do_not_specialize=['ROWS'])
def _forward(A,B,C,Y,ROWS,D:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,SC:tl.constexpr,
         KIND:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    row=(tl.program_id(0)*BR+tl.arange(0,BR))[:,None]
    col=tl.arange(0,BC)[None,:]
    i=row*D+col;mask=(row<ROWS)&(col<D)
    a=tl.load(A+row*SA+col,mask,0).to(tl.float32)
    b=tl.load(B+row*SB+col,mask,0).to(tl.float32)
    if KIND==0:
        tg=tl.math.tanh(b*.25);th=tl.math.tanh(a*.04)
        sigmoid=1/(1+tl.exp(-b))
        y=(4*tg*sigmoid)*(25*th)
    else:
        c=tl.load(C+row*SC+col,mask,0).to(tl.float32)
        if KIND==1:y=a*(1+b)+c
        else:y=a+b*c
    tl.store(Y+i,y,mask)


@triton.jit(do_not_specialize=['ROWS'])
def _backward(A,B,C,G,DA,DB,DC,ROWS,D:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,
          SC:tl.constexpr,SG:tl.constexpr,KIND:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    row=(tl.program_id(0)*BR+tl.arange(0,BR))[:,None]
    col=tl.arange(0,BC)[None,:]
    i=row*D+col;mask=(row<ROWS)&(col<D)
    g=tl.load(G+row*SG+col,mask,0).to(tl.float32)
    if KIND==0:
        a=tl.load(A+row*SA+col,mask,0).to(tl.float32)
        b=tl.load(B+row*SB+col,mask,0).to(tl.float32)
        tg=tl.math.tanh(b*.25);th=tl.math.tanh(a*.04)
        sigmoid=1/(1+tl.exp(-b))
        da=g*(4*tg*sigmoid)*(1-th*th)
        db=g*(25*th)*((1-tg*tg)*sigmoid+4*tg*sigmoid*(1-sigmoid))
        tl.store(DA+i,da,mask);tl.store(DB+i,db,mask)
    elif KIND==1:
        a=tl.load(A+row*SA+col,mask,0).to(tl.float32)
        b=tl.load(B+row*SB+col,mask,0).to(tl.float32)
        tl.store(DA+i,g*(1+b),mask);tl.store(DB+i,g*a,mask)
    else:
        b=tl.load(B+row*SB+col,mask,0).to(tl.float32)
        c=tl.load(C+row*SC+col,mask,0).to(tl.float32)
        tl.store(DB+i,g*c,mask);tl.store(DC+i,g*b,mask)


class _Pointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,c,kind):
        if a.ndim!=2 or b.shape!=a.shape or (kind and c.shape!=a.shape):
            raise ValueError('MOSS fused kernels require matching two-dimensional tensors')
        if a.stride(1)!=1 or b.stride(1)!=1 or (kind and c.stride(1)!=1):
            raise ValueError('MOSS fused kernels require contiguous innermost dimensions')
        if a.dtype!=b.dtype or (kind and a.dtype!=c.dtype):
            raise ValueError('MOSS fused kernels require equal activation dtypes')
        output=torch.empty_like(a,memory_format=torch.contiguous_format)
        ctx.kind=kind;ctx.save_for_backward(a,b,c)
        # Modulation is materialized once before the DiT stack. Its dense
        # affine/residual path needs neither row division nor power-of-two
        # channel padding. Keep strided QKV/FFN slices on the row kernel.
        ctx.dense=bool(kind and all(x.is_contiguous() for x in (a,b,c)))
        if a.numel():
            if ctx.dense:
                _dense_forward[(triton.cdiv(a.numel(),4096),)](a,b,c,output,a.numel(),kind,4096,enable_fp_fusion=False)
            else:
                br=4 if a.shape[1]<1024 else (2 if a.shape[1]<=2048 else 1)
                _forward[(triton.cdiv(a.shape[0],br),)](a,b,c,output,a.shape[0],a.shape[1],
                    a.stride(0),b.stride(0),c.stride(0),kind,br,triton.next_power_of_2(a.shape[1]),enable_fp_fusion=False)
        return output

    @staticmethod
    def backward(ctx,grad):
        a,b,c=ctx.saved_tensors;kind=ctx.kind
        if grad.stride(1)!=1:grad=grad.contiguous()
        da=grad if kind==2 else torch.empty_like(a,memory_format=torch.contiguous_format)
        db=torch.empty_like(b,memory_format=torch.contiguous_format)
        dc=grad if kind==1 else (torch.empty_like(c,memory_format=torch.contiguous_format) if kind==2 else None)
        if a.numel():
            if ctx.dense and grad.is_contiguous():
                _dense_backward[(triton.cdiv(a.numel(),4096),)](a,b,c,grad,da,db,dc,a.numel(),kind,4096,enable_fp_fusion=False)
            else:
                br=4 if a.shape[1]<1024 else (2 if a.shape[1]<=2048 else 1)
                _backward[(triton.cdiv(a.shape[0],br),)](a,b,c,grad,da,db,dc if dc is not None else da,
                    a.shape[0],a.shape[1],a.stride(0),b.stride(0),c.stride(0),grad.stride(0),kind,br,
                    triton.next_power_of_2(a.shape[1]),enable_fp_fusion=False)
        return da,db,dc,None


def situ(hidden,gate):
    return _Pointwise.apply(hidden,gate,hidden,0)


def affine(hidden,scale,shift):
    return _Pointwise.apply(hidden,scale,shift,1)


def gated_add(hidden,branch,gate):
    return _Pointwise.apply(hidden,branch,gate,2)
