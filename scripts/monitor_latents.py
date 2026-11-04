#!/usr/bin/env python3
"""Read-only latent extraction monitoring. No torch imports or tensor reads."""

import argparse
from collections import deque
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import threading
import time
from urllib.parse import urlsplit

RESOLUTIONS = ("832x480", "960x544")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def process_matches(pid, module, value=None, proc_root=Path("/proc")):
    try:
        args = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
        state = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z" and module.encode() in args and (
            value is None or str(value).encode() in args
        )
    except OSError:
        return False


def progress_rows(path, offset=0):
    try:
        with Path(path).open("rb") as handle:
            handle.seek(offset)
            lines = handle.read().decode(errors="replace").splitlines()
    except OSError:
        return [], []
    rows, errors = [], []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            if "Traceback (most recent" in line or "Error:" in line:
                errors.append(line[:350])
            continue
        if isinstance(row, dict) and "sample_id" in row and "completed" in row:
            rows.append(row)
    return rows, errors[-4:]


def seconds_per_pair(rows, window=180):
    """Exclude cache-only replay and model startup from encoding speed."""
    if not rows:
        return None
    end = rows[-1]["elapsed_seconds"]
    times = []
    for a, b in zip(rows, rows[1:]):
        if a["elapsed_seconds"] < end - window:
            continue
        dt = b["elapsed_seconds"] - a["elapsed_seconds"]
        if (b["completed"] == a["completed"] + 1 and dt > 0 and
                all(b["encoded"].get(k, 0) > a["encoded"].get(k, 0)
                    for k in ("qwen", "wan"))):
            times.append(dt)
    return statistics.median(times) if len(times) >= 3 else None


class Monitor:
    def __init__(self, root, interval=10):
        self.root = Path(root).resolve()
        self.folder = self.root / "metadata/latent_validation"
        self.interval = interval
        self.lock = threading.Lock()
        self.snapshot = None
        self.history = deque(maxlen=720)
        self.run_id = None
        self.error = None
        self.manifests = {}

    def manifest(self, resolution):
        path = self.root / resolution / "samples.jsonl"
        stamp = path.stat().st_mtime_ns
        if resolution not in self.manifests or self.manifests[resolution][0] != stamp:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.manifests[resolution] = stamp, rows
        return self.manifests[resolution][1]

    def collect(self):
        now = time.time()
        launch = read_json(self.folder / "latest_restart.json")
        status = read_json(self.folder / "full_status.json")
        offsets = launch.get("log_offsets", {})
        run_id = (launch.get("pid"), launch.get("started_utc"))
        if self.run_id != run_id:
            self.history.clear()
            self.run_id = run_id
        started = datetime.strptime(launch["started_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        devices = status.get("devices", [0, 1, 2, 3])
        processes = {}
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                argv = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
                if "scripts.prepare_latents" not in argv:
                    continue
                manifest = argv[argv.index("--manifest") + 1]
                if Path(manifest).parent.parent != self.root:
                    continue
                shard = int(argv[argv.index("--shard-index") + 1]) if "--shard-index" in argv else 0
                if process_matches(path.name, "scripts.prepare_latents", manifest):
                    processes[(Path(manifest).parent.name, shard)] = int(path.name)
            except (OSError, ValueError, IndexError):
                continue
        workers, resolutions, errors, estimates = [], [], [], {}
        for res in RESOLUTIONS:
            manifest = self.root / res / "samples.jsonl"
            rows = self.manifest(res)
            # Count only branches selected by the manifest's encoder profile.
            profile_paths = {r["latent_profile"] for r in rows if r.get("latent_profile")}
            if len(profile_paths) != 1:
                raise ValueError(f"{res}: expected exactly one active latent profile")
            profile = read_json(manifest.parent / next(iter(profile_paths)))
            from hashlib import sha256
            def digest(value):
                return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            files = {}
            for kind in ("qwen", "wan"):
                preparation = {k:v for k,v in profile["video"].items()
                               if kind == "qwen" or not k.startswith("qwen_")}
                branch = digest({"encoder":profile["encoders"][kind], "preprocessing":preparation})[:20]
                files[kind] = {p.stem for p in (manifest.parent / "latents" / kind / branch).glob("*/*.pt")}
            paired = len(files["qwen"] & files["wan"])
            indexed = sum(bool(r.get("qwen_cache")) and bool(r.get("wan_cache")) for r in rows)
            pilot, _ = progress_rows(self.folder / f"pilot_{res}.log")
            historical_rate = seconds_per_pair(pilot, 100000)
            slow_rates = []
            for shard in range(len(devices)):
                old, _ = progress_rows(self.folder / f"full_parallel_{res}_shard-{shard}.log")
                rate = seconds_per_pair(old)
                if rate:
                    slow_rates.append(rate)
            live_rates = []
            for shard, device in enumerate(devices):
                p = self.folder / f"full_{res}_shard-{shard:02d}.log"
                current, log_errors = progress_rows(p, offsets.get(str(p), 0))
                # A log from an earlier supervisor must never imply current progress.
                if p.exists() and p.stat().st_mtime < started:
                    current, log_errors = [], []
                report = read_json(manifest.parent / "metadata/latent_runs" / f"full.shard-{shard:02d}.json")
                if report.get("started", 0) < started:
                    report = {}
                latest = current[-1] if current else {}
                pid = processes.get((res, shard))
                age = now-p.stat().st_mtime if p.exists() and current else None
                rate = seconds_per_pair(current)
                if rate and pid and age is not None and age < 120:
                    live_rates.append(rate)
                selected = len(range(shard, len(rows), len(devices)))
                complete = report.get("complete", False)
                state = "complete" if complete else (
                    "stalled" if pid and age is not None and age > 120 else
                    "running" if pid else "stopped" if latest or log_errors else "waiting")
                w = {"resolution":res,"shard":shard,"device":device,"pid":pid,
                     "state":state,"completed":latest.get("completed",0),"total":selected,
                     "encoded_qwen":latest.get("encoded",{}).get("qwen",0),
                     "encoded_wan":latest.get("encoded",{}).get("wan",0),
                     "reused_qwen":latest.get("reused",{}).get("qwen",0),
                     "reused_wan":latest.get("reused",{}).get("wan",0),
                     "seconds_per_pair":rate,"log_age_seconds":age,
                     "sample_id":latest.get("sample_id"),"log_name":p.name}
                workers.append(w)
                errors.extend(f"{res} / {shard}: {e}" for e in log_errors)
            rate = statistics.median(live_rates) if live_rates else historical_rate
            estimates[res] = (rate, max([rate or 0, *slow_rates]), bool(live_rates))
            resolutions.append({"resolution":res,"total":len(rows),"qwen":len(files["qwen"]),
                                "wan":len(files["wan"]),"paired":paired,"indexed":indexed,
                                "percentage":100*paired/len(rows),"seconds_per_pair":rate,
                                "rate_source":"live" if live_rates else "historical",
                                "active_workers":sum(bool(w["pid"]) for w in workers if w["resolution"]==res)})
        supervisor_alive = process_matches(launch.get("pid"), "scripts.prepare_openvid81", self.root)
        status_current = status.get("pid") == launch.get("pid")
        done = status_current and status.get("complete",False)
        errors.extend(status.get("failures",[]) if status_current else [])
        if status_current and status.get("error"):
            errors.append(status["error"])
        total = sum(r["total"] for r in resolutions)
        paired = sum(r["paired"] for r in resolutions)
        state = "complete" if done else "failed" if errors else (
            "stopped" if not supervisor_alive and not processes else
            "stalled" if any(w["state"]=="stalled" for w in workers) else
            "verifying" if status_current and status.get("phase")=="verifying" else "running")
        # One worker traverses both resolutions serially; estimate each device's critical path.
        paths=[]
        for shard in range(len(devices)):
            low=high=0
            for r in resolutions:
                w=next(w for w in workers if w["resolution"]==r["resolution"] and w["shard"]==shard)
                rem=max(0,w["total"]-w["completed"]) if w["completed"] else max(0,r["total"]-r["paired"])/len(devices)
                rate,slow,live=estimates[r["resolution"]]
                if rem and not rate:
                    low=high=float('inf')
                    break
                low+=rem*(rate or 0)
                high+=rem*((rate or 0)*1.15 if live else slow)
            paths.append((low,high))
        eta=None
        if state=="running" and all(v[1]!=float('inf') for v in paths):
            eta={"low_seconds":max(v[0] for v in paths)+300,"high_seconds":max(v[1] for v in paths)+600,
                 "includes_verification":True,"verification_allowance_seconds":[300,600]}
        active=[w for w in workers if w["pid"]]
        throughput=sum(60/w["seconds_per_pair"] for w in active if w["seconds_per_pair"] and w["state"]=="running")
        point={"time":now,"paired":paired,"qwen":sum(r["qwen"] for r in resolutions),
               "wan":sum(r["wan"] for r in resolutions),"pairs_per_minute":throughput}
        self.history.append(point)
        result={"updated_at":now,"started_at":started,"state":state,"supervisor_pid":launch.get("pid"),
                "supervisor_alive":supervisor_alive,"total":total,"paired":paired,
                "active_workers":len(active),"expected_workers":len(devices),"pairs_per_minute":throughput,
                "resolutions":resolutions,"workers":workers,"eta":eta,"errors":errors[-8:],
                "history":list(self.history),"poll_seconds":self.interval,
                "sources":{"root":str(self.root),"status":str(self.folder/'full_status.json'),
                           "definition":"paired = intersection of committed Qwen and Wan cache keys in the active profile; indexed = manifest rows with both references"}}
        with self.lock:
            self.snapshot=result
            self.error=None
        return result

    def loop(self):
        while True:
            try:
                self.collect()
            except Exception as exc:
                with self.lock:
                    self.error=f"{type(exc).__name__}: {exc}"
            time.sleep(self.interval)

    def get(self):
        with self.lock:
            return {**(self.snapshot or {}),"monitor_error":self.error}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,monitor,**kwargs):
        self.monitor=monitor
        super().__init__(*args,**kwargs)

    def do_GET(self):
        if urlsplit(self.path).path=="/api/live":
            body=json.dumps(self.monitor.get(),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self,*args):
        pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--web-root")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--interval",type=int,default=10)
    p.add_argument("--snapshot",action="store_true")
    args=p.parse_args()
    monitor=Monitor(args.root,max(5,args.interval))
    if args.snapshot:
        print(json.dumps(monitor.collect(),ensure_ascii=False,indent=2))
        return
    if not args.web_root:
        p.error("--web-root is required for the server")
    monitor.collect()
    threading.Thread(target=monitor.loop,daemon=True).start()
    server=ThreadingHTTPServer((args.host,args.port),partial(Handler,monitor=monitor,directory=args.web_root))
    print(f"Latent monitor: http://{args.host}:{args.port}/",flush=True)
    server.serve_forever()


if __name__=="__main__":
    main()
