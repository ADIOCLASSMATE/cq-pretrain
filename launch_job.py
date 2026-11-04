"""Fixed-size TP x DP launcher using the 64-card sweep's rendezvous protocol."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

from utils.network import rendezvous_interface
from utils.distributed import atomic_json, wait_files

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rendezvous', required=True, type=Path, help='Unique shared launch directory')
    parser.add_argument('--nodes', type=int, default=int(os.environ.get('PET_NNODES', '1')))
    parser.add_argument('--node-rank', type=int, default=int(os.environ.get('PET_NODE_RANK', '0')))
    parser.add_argument('--nproc-per-node', type=int, default=16)
    parser.add_argument('--hccl-base-port', type=int, default=28000)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    if args.nodes < 1 or not 0 <= args.node_rank < args.nodes or not train_args:
        raise ValueError('Require a valid node topology and training arguments after --')
    if not 1024 <= args.hccl_base_port <= 65472:
        raise ValueError('Invalid HCCL base port')
    ephemeral = list(map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split()))
    if args.hccl_base_port <= ephemeral[1] and args.hccl_base_port+31 >= ephemeral[0]:
        raise ValueError('Choose an HCCL range outside the host ephemeral TCP range')
    command = ['bash', str(ROOT/'scripts/launch.sh'), *train_args]
    if args.dry_run:
        print(json.dumps(dict(world=args.nodes*args.nproc_per_node, nodes=args.nodes,
            node_rank=args.node_rank, hccl_base_port=args.hccl_base_port,
            command=command, rendezvous=str(args.rendezvous)), indent=2))
        return
    master = os.environ.get('PET_MASTER_ADDR') or os.environ.get('MASTER_ADDR', '127.0.0.1')
    port = os.environ.get('PET_MASTER_PORT') or os.environ.get('MASTER_PORT', '29500')
    interface, addresses = rendezvous_interface(master)
    receipt = args.rendezvous/f'node{args.node_rank}.json'
    if receipt.exists():
        raise FileExistsError('Use a new rendezvous directory for every launch/resume')
    from utils.experiment import source_hashes
    sources=source_hashes()
    atomic_json(receipt, dict(hostname=socket.gethostname(), ip=addresses[interface],
                interface=interface, source_sha256=sources, nproc=args.nproc_per_node))
    files = [args.rendezvous/f'node{n}.json' for n in range(args.nodes)]
    wait_files(files, 600)
    peers = [json.loads(p.read_text()) for p in files]
    if any(p['source_sha256'] != sources or p['nproc'] != args.nproc_per_node for p in peers):
        raise ValueError('All nodes must use the same sources and local world size')
    env = dict(os.environ, NNODES=str(args.nodes), NODE_RANK=str(args.node_rank),
        NPROC_PER_NODE=str(args.nproc_per_node), MASTER_ADDR=peers[0]['ip'], MASTER_PORT=port,
        GLOO_SOCKET_IFNAME=interface, HCCL_SOCKET_IFNAME='='+interface,
        HCCL_IF_IP=addresses[interface], HCCL_IF_BASE_PORT=str(args.hccl_base_port),
        PRETRAIN_PYTHON=sys.executable, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'),
        TORCH_EXTENSIONS_DIR=str(ROOT/'.deps/extensions-core17'))
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE'):
        env.pop(key, None)
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
