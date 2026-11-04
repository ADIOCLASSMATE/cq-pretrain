"""Exact model-only TP conversion; optimizer resume remains topology-strict."""
import torch


def reshard_tensor(values, target, target_tp, target_rank):
    """Read the requested logical slice, including fused strided projections.

    ``values`` contains all saved TP slices in TP-rank order. Metadata is taken
    from the constructed destination Parameter, not guessed from its name.
    No floating-point arithmetic or dtype conversion is involved.
    """
    source_tp=len(values)
    if not source_tp or not 0<=target_rank<target_tp:
        raise ValueError('Invalid source/destination tensor-parallel topology')
    if not all(isinstance(v,torch.Tensor) for v in values):
        raise TypeError('Tensor checkpoint entry required')
    if any(v.dtype!=target.dtype for v in values):
        raise ValueError('Checkpoint resharding must preserve parameter dtype')
    if not getattr(target,'tensor_model_parallel',False):
        if any(v.shape!=target.shape or not torch.equal(v,values[0]) for v in values):
            raise ValueError('Replicated checkpoint tensors differ across TP ranks')
        return values[0]
    dim=target.partition_dim
    global_width=target.shape[dim]*target_tp
    splits=getattr(target,'_checkpoint_splits',None)
    if splits is None:
        stride=getattr(target,'partition_stride',1)
        if global_width%stride:raise ValueError('Invalid projection stride')
        splits=(global_width//stride,)*stride
    if sum(splits)!=global_width or any(n%source_tp or n%target_tp for n in splits):
        raise ValueError('Logical projection widths do not divide both TP sizes')
    shape=list(target.shape);shape[dim]=global_width//source_tp
    if any(list(v.shape)!=shape for v in values):
        raise ValueError('Saved TP shape does not match destination metadata')
    source_parts=[v.split([n//source_tp for n in splits],dim=dim) for v in values]
    destination=[]
    for part,width in enumerate(splits):
        begin=target_rank*(width//target_tp);end=begin+width//target_tp
        slices=[]
        for rank in range(source_tp):
            origin=rank*(width//source_tp)
            left=max(begin,origin);right=min(end,origin+width//source_tp)
            if left<right:slices.append(source_parts[rank][part].narrow(dim,left-origin,right-left))
        destination.append(slices[0] if len(slices)==1 else torch.cat(slices,dim=dim))
    result=destination[0] if len(destination)==1 else torch.cat(destination,dim=dim)
    if result.shape!=target.shape:raise ValueError('Unexpected resharded tensor shape')
    return result


def reshard_state(states, model, target_tp, target_rank):
    target=model.state_dict(keep_vars=True)
    if any(state.keys()!=target.keys() for state in states):
        raise ValueError('Initialization checkpoint keys differ from model')
    output={}
    for name,parameter in target.items():
        values=[state[name] for state in states]
        if parameter is None:
            if any(v is not None for v in values):raise ValueError(f'Unexpected extra state: {name}')
            output[name]=None
        elif isinstance(parameter,torch.Tensor):
            try:output[name]=reshard_tensor(values,parameter,target_tp,target_rank)
            except (ValueError,TypeError) as error:raise ValueError(f'{name}: {error}') from error
        else:raise TypeError(f'Unsupported checkpoint entry: {name}')
    return output
