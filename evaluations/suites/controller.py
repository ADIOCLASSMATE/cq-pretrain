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
again (it resumes from its partial outputs) up to job.max_attempts times. Submission
stops early, with an ATTENTION file, once two attempts in a row ended without adding
any output file: that is a failure, not a preemption.
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


def count_files(path):
    return sum(len(files) for _, _, files in os.walk(path))


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
        self.capacity, self.quotas = {}, {}
        cli = args.cli or (str(Path.home() / '.local/bin/inspire') if (Path.home() / '.local/bin/inspire').exists()
                           else 'inspire')
        self.base = [cli, '--json'] + (['--account', args.account] if args.account else [])
        self.cli = [*self.base, 'job']
        (self.root / 'jobs').mkdir(parents=True, exist_ok=True)

    def log(self, message, **fields):
        line = json.dumps(dict(time=now(), message=message, **fields), ensure_ascii=False)
        with (self.root / 'controller.log').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)

    def call(self, *arguments, timeout=180, group='job'):
        command = [*self.base, group] if group != 'job' else self.cli
        for attempt in range(5):
            process = subprocess.run([*command, *arguments], capture_output=True, text=True, timeout=timeout)
            try:
                reply = json.loads(process.stdout or process.stderr)
            except ValueError:
                reply = {'success': False, 'error': (process.stdout or process.stderr)[-800:]}
            if reply.get('success'):
                return reply['data']
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f'inspire {" ".join(arguments[:2])} failed: {str(reply)[:800]}')

    def observe_capacity(self):
        """Free GPUs and whole free nodes per eligible group, read once per cycle."""
        j = self.job
        if not self.quotas:
            rows = self.call('quota', '--workspace', j['workspace'], '--all')['items']
            self.quotas = {r['compute_group']: r['quota'] for r in rows
                           if r['quota'].split(',')[0] == str(j['gpus'])}
        items = self.call('availability', '--workspace', j['workspace'], group='resources')['items']
        self.capacity = {}
        for item in items:
            name = item['compute_group']
            if (name.startswith(j['zone']) and name in self.quotas
                    and any(t in item['gpu_type'] for t in j['gpu_types'])):
                per_node = max(1, int(item.get('gpus_per_node') or 8) // int(j['gpus']))
                self.capacity[name] = dict(free=int(item['available_gpus']),
                                           slots=int(item['full_free_nodes']) * per_node, gpu=item['gpu_type'])

    def place(self):
        """The group to submit to now, or None when no eligible group has room."""
        gpus = int(self.job['gpus'])
        ranked = sorted(self.capacity.items(), key=lambda kv: (kv[1]['slots'] > 0, kv[1]['free']), reverse=True)
        if not ranked or ranked[0][1]['free'] < gpus:
            return None
        name, room = ranked[0]
        room['free'] -= gpus
        room['slots'] = max(0, room['slots'] - 1)
        return name

    def records(self):
        return [json.loads(p.read_text()) for p in sorted((self.root / 'jobs').glob('*.json'))]

    def refresh(self, record):
        if record.get('status', '').upper() in TERMINAL or self.dry:
            return record
        waited = (datetime.now(timezone.utc) - datetime.fromisoformat(record['submitted'])).total_seconds() / 60
        if record.get('status', '').upper() in ('QUEUING', 'PENDING', 'SUBMITTED') and \
                waited > float(self.job.get('queue_timeout_minutes', 20)):
            # Still waiting for cards: withdraw it so the next placement can use
            # a group that has room now. Its outputs, if any, are kept.
            try:
                self.call('stop', record['name'], '--workspace', self.job['workspace'])
            except RuntimeError as error:
                self.log('withdraw failed', name=record['name'], error=str(error)[:300])
                return record
            record.update(status='STOPPED', reason='queue_timeout', observed=now())
            write_json(self.root / 'jobs' / f'{record["name"]}.json', record)
            self.log('withdrawn after queueing', name=record['name'], minutes=round(waited, 1), group=record.get('group'))
            return record
        try:
            data = self.call('status', record['name'], '--workspace', self.job['workspace'])
        except RuntimeError as error:
            self.log('status unavailable', name=record['name'], error=str(error)[:300])
            return record
        if data.get('status') != record.get('status'):
            self.log('status', name=record['name'], status=data.get('status'), priority=data.get('priority_level'))
            if str(data.get('status', '')).upper() in TERMINAL:
                record['files_at_end'] = count_files(record['output'])
        record.update(status=data.get('status'), priority_level=data.get('priority_level'), observed=now())
        write_json(self.root / 'jobs' / f'{record["name"]}.json', record)
        return record

    def submit(self, task, step, snapshot, attempt, group):
        name = f'{self.prefix}-{task}-s{step}-r{attempt:02d}'
        out = task_dir(self.root, task, step)
        command = shlex.join(['bash', str(self.source / 'scripts/evaluate.sh'), task, str(snapshot), str(out)])
        j = self.job
        arguments = ['create', '--name', name, '--workspace', j['workspace'], '--project', j['project'],
                     '--group', group, '--quota', self.quotas[group], '--nodes', '1', '--image', j['image'],
                     '--priority', str(j['priority']), '--shm-size', str(j['shm_size']),
                     # No platform auto-restart: a deterministic failure would loop in
                     # place. This controller resubmits and stops when nothing progresses.
                     '--env', f'NPROC_PER_NODE={j["gpus"]}',
                     '--env', f'TRITON_CACHE_DIR={self.root / ".triton"}', '--command', command]
        record = dict(name=name, task=task, step=step, attempt=attempt, snapshot=str(snapshot), output=str(out),
                      group=group, quota=self.quotas[group], command=command, submitted=now(), status='SUBMITTED',
                      files_at_submit=count_files(out))
        if self.dry:
            self.log('dry-run submit', **record)
            return record
        dry = self.call(*arguments, '--dry-run')
        if dry.get('resource', {}).get('gpu') != j['gpus']:
            raise RuntimeError(f'Unexpected evaluation resources: {dry.get("resource")}')
        created = self.call(*arguments)
        record.update(created=created)
        write_json(self.root / 'jobs' / f'{name}.json', record)
        self.log('submitted', name=name, task=task, step=step, attempt=attempt, group=group)
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
        self.observe_capacity()
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
                ran = [r for r in sorted(mine, key=lambda r: r['attempt']) if r.get('reason') != 'queue_timeout']
                stalled = len(ran) >= 2 and all('files_at_end' in r and r['files_at_end'] <= r.get('files_at_submit', 0)
                                                for r in ran[-2:])
                if stalled:
                    attention = self.root / f'ATTENTION-{task}-s{step}.json'
                    if not attention.exists():
                        write_json(attention, dict(time=now(), task=task, step=step, reason='no progress in two attempts',
                                                   jobs=[r['name'] for r in ran[-2:]]))
                        self.log('stopped: no progress in two attempts', task=task, step=step)
                    continue
                if attempt >= self.job['max_attempts']:
                    attention = self.root / f'ATTENTION-{task}-s{step}.json'
                    if not attention.exists():
                        write_json(attention, dict(time=now(), task=task, step=step, jobs=[r['name'] for r in mine]))
                        self.log('attempts exhausted', task=task, step=step)
                    continue
                if running >= self.job['max_running']:
                    continue
                group = self.place()
                if group is None:
                    self.log('no free cards', task=task, step=step,
                             capacity={k: v['free'] for k, v in self.capacity.items()})
                    continue
                record = self.submit(task, step, snapshot, attempt, group)
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
