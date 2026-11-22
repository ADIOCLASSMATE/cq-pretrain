"""Submit, follow and collect benchmark jobs for every committed evaluation snapshot.

Runs on a CPU host that has the inspire CLI; it needs no GPU and no torch:
    .venv-cuda/bin/python -m evaluations.suites.controller --run RUN --source SRC
State lives under RUN/benchmarks:
    jobs/<name>.json            the submission and its last observed platform status
    <task>/step_<step>/         resumable task outputs; RESULTS.json once scored
    RESULTS.md, results.jsonl   the score table, rewritten every cycle
    controller.log              one line per decision
A step's snapshot is deleted once all of its tasks have RESULTS.json (unless --keep).
Low-priority jobs can be preempted; a job that ends without RESULTS.json is submitted
again (it resumes from its partial outputs) up to job.max_attempts times.
Create RUN/benchmarks/STOP to stop submitting; running jobs are left alone.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from evaluations.suites import settings, task_dir

TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'STOPPED', 'JOB_STOPPED', 'TERMINATED', 'COMPLETED', 'DELETED'}
COLUMNS = (('GQA ↑', 'qa', lambda r: 100 * r['metrics']['gqa_accuracy'], '{:.2f}'),
           ('MME 感知 ↑', 'qa', lambda r: r['metrics']['mme_perception_score'], '{:.2f}'),
           ('MME 认知 ↑', 'qa', lambda r: r['metrics']['mme_cognition_score'], '{:.2f}'),
           ('POPE F1 ↑', 'qa', lambda r: 100 * r['metrics']['pope_macro_f1'], '{:.2f}'),
           ('TextVQA ↑', 'qa', lambda r: 100 * r['metrics']['textvqa_vqa_accuracy'], '{:.2f}'),
           ('VQAv2 ↑', 'vqav2', lambda r: r['metrics']['vqav2_accuracy_pct'], '{:.2f}'),
           ('FID-50K ↓', 'fid', lambda r: r['FID'], '{:.2f}'),
           ('GenEval ↑', 'geneval', lambda r: 100 * r['overall'], '{:.2f}'))


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def committed(path):
    """A complete evaluation snapshot, checked without torch (see utils.eval_snapshot)."""
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
    except (OSError, ValueError):
        return None
    if manifest.get('kind') != 'inference_weights' or path.is_symlink():
        return None
    for name, size in manifest.get('files', {}).items():
        shard = path / name
        if not shard.is_file() or shard.stat().st_size != size:
            return None
    return manifest


class Controller:
    def __init__(self, args):
        self.run, self.source = args.run.resolve(), args.source.resolve()
        self.options = settings(args.config)
        self.job = self.options['job']
        self.tasks = self.options['tasks']
        self.root = self.run / 'benchmarks'
        self.snapshots = args.snapshots.resolve() if args.snapshots else self.run / 'eval_snapshots'
        self.prefix, self.keep, self.dry = args.prefix, args.keep, args.dry_run
        cli = args.cli or (str(Path.home() / '.local/bin/inspire') if (Path.home() / '.local/bin/inspire').exists()
                           else 'inspire')
        self.cli = [cli, '--json', 'job'] + (['--account', args.account] if args.account else [])
        (self.root / 'jobs').mkdir(parents=True, exist_ok=True)

    def log(self, message, **fields):
        line = json.dumps(dict(time=now(), message=message, **fields), ensure_ascii=False)
        with (self.root / 'controller.log').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)

    def call(self, *arguments, timeout=180):
        for attempt in range(5):
            process = subprocess.run([*self.cli, *arguments], capture_output=True, text=True, timeout=timeout)
            try:
                reply = json.loads(process.stdout or process.stderr)
            except ValueError:
                reply = {'success': False, 'error': (process.stdout or process.stderr)[-800:]}
            if reply.get('success'):
                return reply['data']
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f'inspire {" ".join(arguments[:2])} failed: {str(reply)[:800]}')

    def records(self):
        return [json.loads(p.read_text()) for p in sorted((self.root / 'jobs').glob('*.json'))]

    def refresh(self, record):
        if record.get('status', '').upper() in TERMINAL or self.dry:
            return record
        try:
            data = self.call('status', record['name'], '--workspace', self.job['workspace'])
        except RuntimeError as error:
            self.log('status unavailable', name=record['name'], error=str(error)[:300])
            return record
        if data.get('status') != record.get('status'):
            self.log('status', name=record['name'], status=data.get('status'), priority=data.get('priority_level'))
        record.update(status=data.get('status'), priority_level=data.get('priority_level'), observed=now())
        write_json(self.root / 'jobs' / f'{record["name"]}.json', record)
        return record

    def submit(self, task, step, snapshot, attempt):
        name = f'{self.prefix}-{task}-s{step}-r{attempt:02d}'
        out = task_dir(self.root, task, step)
        command = shlex.join(['bash', str(self.source / 'scripts/evaluate.sh'), task, str(snapshot), str(out)])
        j = self.job
        arguments = ['create', '--name', name, '--workspace', j['workspace'], '--project', j['project'],
                     '--group', j['group'], '--quota', str(j['quota']), '--nodes', '1', '--image', j['image'],
                     '--priority', str(j['priority']), '--shm-size', str(j['shm_size']),
                     '--auto-fault-tolerance', '--fault-tolerance-max-retry', '100',
                     '--fault-tolerance-retry-interval', '60', '--env', f'NPROC_PER_NODE={j["gpus"]}',
                     '--env', f'TRITON_CACHE_DIR={self.root / ".triton"}', '--command', command]
        record = dict(name=name, task=task, step=step, attempt=attempt, snapshot=str(snapshot), output=str(out),
                      command=command, submitted=now(), status='SUBMITTED')
        if self.dry:
            self.log('dry-run submit', **record)
            return record
        dry = self.call(*arguments, '--dry-run')
        if dry.get('resource', {}).get('gpu') != j['gpus']:
            raise RuntimeError(f'Unexpected evaluation resources: {dry.get("resource")}')
        created = self.call(*arguments)
        record.update(created=created)
        write_json(self.root / 'jobs' / f'{name}.json', record)
        self.log('submitted', name=name, task=task, step=step, attempt=attempt)
        return record

    def table(self):
        rows = []
        for task in self.tasks:
            for path in sorted((self.root / task).glob('step_*/RESULTS.json')):
                rows.append((int(path.parent.name.split('_')[1]), task, json.loads(path.read_text())))
        steps = sorted({step for step, _, _ in rows})
        by = {(step, task): value for step, task, value in rows}
        lines = ['# 训练中评测', '', f'更新时间 {now()}。每个数值覆盖该项的完整评测集；评测完成后对应快照已删除。', '',
                 '| Step | ' + ' | '.join(c[0] for c in COLUMNS) + ' |', '| ---: |' + ' ---: |' * len(COLUMNS)]
        records = []
        for step in steps:
            cells, record = [], dict(step=step)
            for header, task, extract, form in COLUMNS:
                value = by.get((step, task))
                if value is None:
                    cells.append('—')
                    continue
                number = float(extract(value))
                if not math.isfinite(number):
                    raise ValueError(f'Non-finite {header} at step {step}')
                cells.append(form.format(number))
                record[header] = number
            lines.append(f'| {step} | ' + ' | '.join(cells) + ' |')
            records.append(record)
        lines += ['', 'GQA test-dev balanced 12,578 题，MME 2,374 题，POPE 9,000 题（三个划分的宏平均 F1），'
                  'TextVQA val 5,000 题，VQAv2 val 214,354 题，均为单图零样本贪心解码。FID-50K 为 1,000 个 ImageNet 类'
                  '各 50 张 512×512 图像，对照 ADM ImageNet-512 参考统计量；GenEval 为 553 个官方提示词各 4 张图。'
                  '生图统一 30 步 Euler、CFG 3.0、shift 3.0；骨干为在线权重，渲染器为 EMA。', '']
        (self.root / 'RESULTS.md').write_text('\n'.join(lines))
        (self.root / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))

    def cycle(self):
        records = [self.refresh(r) for r in self.records()]
        active = [r for r in records if r.get('status', '').upper() not in TERMINAL]
        running = len(active)
        stop = (self.root / 'STOP').exists()
        for snapshot in sorted(self.snapshots.glob('step_*')):
            manifest = committed(snapshot)
            if manifest is None:
                continue
            step = manifest['step']
            done = True
            for task in self.tasks:
                if (task_dir(self.root, task, step) / 'RESULTS.json').is_file():
                    continue
                done = False
                mine = [r for r in records if r['task'] == task and r['step'] == step]
                if any(r in active for r in mine) or stop:
                    continue
                attempt = 1 + max((r['attempt'] for r in mine), default=-1)
                if attempt >= self.job['max_attempts']:
                    attention = self.root / f'ATTENTION-{task}-s{step}.json'
                    if not attention.exists():
                        write_json(attention, dict(time=now(), task=task, step=step, jobs=[r['name'] for r in mine]))
                        self.log('attempts exhausted', task=task, step=step)
                    continue
                if running >= self.job['max_running']:
                    continue
                record = self.submit(task, step, snapshot, attempt)
                records.append(record)
                active.append(record)
                running += 1
            if done and not self.keep and not self.dry:
                if snapshot.parent.resolve() != self.snapshots.resolve() or snapshot.is_symlink():
                    raise ValueError(f'Refusing to delete {snapshot}')
                shutil.rmtree(snapshot)
                self.log('snapshot removed', step=step, path=str(snapshot))
        self.table()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='Training run directory')
    parser.add_argument('--source', type=Path, required=True, help='Frozen source snapshot the jobs execute')
    parser.add_argument('--snapshots', type=Path, help='Defaults to RUN/eval_snapshots')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--prefix', default='cqeval')
    parser.add_argument('--cli')
    parser.add_argument('--account')
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--keep', action='store_true', help='Keep snapshots after evaluation')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    controller = Controller(args)
    with (controller.root / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller.log('start', pid=os.getpid(), source=str(controller.source), snapshots=str(controller.snapshots))
        while True:
            try:
                controller.cycle()
            except Exception as error:
                controller.log('cycle failed', error=repr(error)[:800])
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
