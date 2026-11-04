"""Unified corpus coordinator, workers and resource-manager configuration."""
import argparse
import json
from pathlib import Path
import time
from .config import STAGES


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','run','status','worker','node-worker','fleet','stop'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--local',action='store_true',help='CPU execution for bounded validation only')
    parser.add_argument('--watch',action='store_true',help='Resume coordinator as remote workers commit tasks')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--queue-backend',choices=('local','rpc'),default='rpc')
    parser.add_argument('--max-tasks',type=int)
    parser.add_argument('--template');parser.add_argument('--output')
    args=parser.parse_args(argv)
    if args.command in ('run','fleet'):
        raw=json.loads(Path(args.config).read_text())
        if raw.get('scope')=='full' and raw.get('policy',{}).get('profile')!='task_aware_v1':
            parser.error('New full runs require task_aware_v1; use the maintained prepare entry. Resume older runs with their frozen runtime.')
    from . import workflow
    if args.command=='preflight':result=workflow.preflight(args.config)
    elif args.command=='status':result=workflow.status(args.config)
    elif args.command=='stop':
        from .fleet import request_stop
        result=request_stop(args.config)
    elif args.command=='run':
        while True:
            result=workflow.run(args.config,local=args.local,stop_after=args.stage)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            if not args.watch or result.get('phase')!='waiting':return 0 if result.get('complete') else 2
            time.sleep(30)
    elif args.command in ('worker','node-worker'):
        if not args.stage:parser.error('worker requires --stage')
        from . import worker
        if args.command=='worker':result=worker.run(args.config,args.stage,device=args.device,
            queue_backend=args.queue_backend,max_tasks=args.max_tasks)
        else:return worker.node_worker(args.config,args.stage,queue_backend=args.queue_backend)
    else:
        if not args.template:parser.error('fleet requires --template')
        from .fleet import build
        result=build(args.config,args.template,output=args.output)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if args.command!='preflight' or result['ready'] else 2


if __name__=='__main__':raise SystemExit(main())
