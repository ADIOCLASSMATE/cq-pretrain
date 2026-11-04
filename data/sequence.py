"""Native video timestamps/markers with frame-block causal CQ content ranks."""
import numpy as np
import torch


def native_video_metadata(timestamps):
    from transformers.video_utils import VideoMetadata
    return VideoMetadata(total_num_frames=len(timestamps),fps=1.0,frames_indices=list(timestamps))


def video_sequence(processor,grid,timestamps):
    # Same serialization as the pinned Qwen3VLProcessor, without patchifying twice.
    times=processor._calculate_timestamps(list(timestamps),1.,processor.video_processor.temporal_patch_size)
    frame_tokens=grid[1]*grid[2]//processor.video_processor.merge_size**2
    text=''.join(f'<{time:.1f} seconds>'+processor.vision_start_token+
                 processor.video_token*frame_tokens+processor.vision_end_token for time in times)
    encoded=processor.tokenizer(text,add_special_tokens=False)['input_ids']
    ids=np.asarray(encoded,dtype=np.int64)
    visual=np.flatnonzero(ids==processor.video_token_id)
    t,h,w=grid;h//=2;w//=2
    if len(visual)!=t*h*w:raise ValueError('Native video placeholders differ from the merger grid')
    groups=visual.reshape(t,h*w)
    sigma=np.arange(len(ids),dtype=np.int64)
    positions=np.zeros((3,len(ids),1),dtype=np.float32)
    cursor=0;last=0
    for indices in groups:
        start,stop=int(indices[0]),int(indices[-1])+1
        if stop-start!=h*w:raise ValueError('Each native time block must be contiguous')
        positions[:,last:start,0]=np.arange(cursor,cursor+start-last)
        cursor+=start-last
        local=np.arange(h*w)
        positions[0,indices,0]=cursor
        positions[1,indices,0]=cursor+local//w
        positions[2,indices,0]=cursor+local%w
        sigma[indices]=start
        cursor+=max(h,w);last=stop
    positions[:,last:,0]=np.arange(cursor,cursor+len(ids)-last)
    return dict(ids=ids,visual=visual,sigma=sigma,positions=positions,
                # Native markers/time strings are structural inputs, not corpus captions.
                text_indices=np.empty(0,dtype=np.int64))


def image_sequence(grid,tokens):
    _,h,w=grid;h//=2;w//=2;n=h*w
    ids=np.full(n+2,tokens['image'],dtype=np.int64);ids[0]=tokens['boi'];ids[-1]=tokens['eoi']
    positions=np.zeros((3,n+2,1),dtype=np.float32);local=np.arange(n)
    positions[0,1:-1,0]=1;positions[1,1:-1,0]=1+local//w;positions[2,1:-1,0]=1+local%w
    positions[:,-1,0]=1+max(h,w)
    sigma=np.arange(n+2,dtype=np.int64);sigma[1:-1]=1
    return dict(ids=ids,visual=np.arange(1,n+1),sigma=sigma,positions=positions,text_indices=np.array([0]))
