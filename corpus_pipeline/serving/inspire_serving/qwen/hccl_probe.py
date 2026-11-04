"""Check every card and collective before loading 807 GB of model weights."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch_npu
import torch.distributed as dist

rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
torch.npu.set_device(local)
dist.init_process_group('hccl', timeout=timedelta(seconds=480))
world = dist.get_world_size()
assert world == int(os.environ.get('QWEN_EXPECTED_CARDS', '32'))
t = torch.full((1024 * 1024,), rank + 1, dtype=torch.float32, device=f'npu:{local}')
dist.all_reduce(t)
assert bool(torch.all(t == world * (world + 1) // 2).item())
v = torch.tensor([rank], dtype=torch.int32, device=f'npu:{local}')
gathered = [torch.zeros_like(v) for _ in range(world)]
dist.all_gather(gathered, v)
assert [int(x.item()) for x in gathered] == list(range(world))
dist.barrier()
props = torch.npu.get_device_properties(local)
report = {'rank': rank, 'local_rank': local, 'world_size': world, 'hostname': socket.gethostname(),
          'device_name': str(props.name), 'total_memory_bytes': props.total_memory,
          'correctness_passed': True, 'time': time.time(), 'boot_id': os.environ['QWEN_BOOT_ID']}
p = Path(os.environ['QWEN_RUN_DIR']) / f'hccl-rank-{rank:02}.json'
p.write_text(json.dumps(report, indent=2) + '\n')
dist.barrier()
dist.destroy_process_group()
if local == 0:
    print(json.dumps({'hccl': 'passed', 'world': world}), flush=True)
