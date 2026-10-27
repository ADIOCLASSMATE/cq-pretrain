"""Install the CQ-pinned runtime locally, without replacing system CANN.

Run with --sources, --python, --cann (independently or together). Use
--host-tools on Ubuntu to install the C++ compiler required by Triton. The
first Python install preserves the previous .venv-npu next to the new environment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "configs/deps.core17.lock.json").read_text())
DEPS = ROOT / ".deps"


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def host_tools():
    """Install host compilation tools only when explicitly requested."""
    if os.geteuid() != 0 or not shutil.which("apt-get"):
        raise RuntimeError("Install a host C++ compiler and make (Ubuntu: sudo apt-get install build-essential)")
    run("apt-get", "update")
    run(
        "apt-get", "install", "-y", "--no-install-recommends",
        *LOCK["gdn_runtime"]["host_build"]["ubuntu_packages"],
        env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"),
    )


def check_host_compiler():
    # Triton-Ascend uses CC as its C++ compiler override.
    compiler = os.environ.get("CC") or shutil.which("clang++") or shutil.which("g++")
    if not compiler or not shutil.which(compiler) or not shutil.which("make"):
        raise RuntimeError(
            "Triton-Ascend needs a host C++ compiler and make. "
            "Install build-essential, or run setup_mindspeed.py --host-tools as root."
        )


def sources():
    for name, entry in LOCK["sources"].items():
        path = DEPS / "core17" / name
        if not (path / ".git").exists():
            path.mkdir(parents=True, exist_ok=True)
            run("git", "-C", path, "init", "-q")
            run("git", "-C", path, "remote", "add", "origin", entry["url"])
        exists = subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", entry["commit"]],
            capture_output=True,
        )
        if exists.returncode:
            run("git", "-C", path, "fetch", "--depth", "1", "origin", entry["commit"])
        run("git", "-C", path, "diff", "--quiet", "--")
        run("git", "-C", path, "checkout", "--detach", entry["commit"])
        run("git", "-C", path, "diff", "--exit-code", "HEAD", "--")
        print(f"Verified {name}: {entry['commit']}", flush=True)


def python():
    check_host_compiler()
    version = LOCK["runtime"]["python"]
    run("uv", "python", "install", version, "--install-dir", ROOT / ".python")
    interpreter = ROOT / f".python/cpython-{version}-linux-x86_64-gnu/bin/python3.12"
    venv = ROOT / ".venv-npu"
    marker = venv / ".umm-core17"
    if venv.exists() and not marker.exists():
        backup = ROOT / f".venv-npu-before-core17-{time.strftime('%Y%m%dT%H%M%S')}"
        venv.rename(backup)
        print(f"Previous environment preserved at {backup}", flush=True)
    if not venv.exists():
        run("uv", "venv", "--relocatable", "--python", interpreter, venv)
        marker.write_text("Core 0.17 runtime; see configs/deps.core17.lock.json\n")
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    run(
        "uv",
        "pip",
        "install",
        "--python",
        venv / "bin/python",
        "--index-strategy",
        "unsafe-best-match",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "-r",
        ROOT / "requirements-mindspeed.txt",
        env=env,
    )
    for name in ("MegatronAdaptor", "TransformerEngineNPU", "MindSpeed"):
        run(
            "uv",
            "pip",
            "install",
            "--python",
            venv / "bin/python",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            DEPS / "core17" / name,
            env=env,
        )
    # Both distributions own files below triton/. Install Ascend LAST, even
    # when rerunning setup, so a community Triton reinstall cannot replace the
    # NPU driver. The wheel's NumPy pin conflicts with MindSpeed's; the exact
    # exception is recorded in the lock instead of modifying vendor metadata.
    ascend = LOCK["gdn_runtime"]["triton_ascend"]
    run(
        venv / "bin/python", "-m", "pip", "install", "--no-deps", "--force-reinstall",
        f"{ascend['url']}#sha256={ascend['sha256']}", env=env,
    )
    run(venv / "bin/python", ROOT / "utils/triton_integrity.py", env=env)
    check = subprocess.run(
        [str(venv / "bin/python"), "-m", "pip", "check"],
        text=True, capture_output=True, env=env,
    )
    known = "triton-ascend 3.2.1 has requirement numpy==1.26.4, but you have numpy 1.26.0."
    problems = check.stdout.strip().splitlines()
    if check.returncode and problems != [known]:
        raise RuntimeError(f"Unexpected dependency conflicts:\n{check.stdout}{check.stderr}")
    if check.returncode:
        print(f"Known upstream dependency metadata exception (see lock): {known}", flush=True)


def cann():
    downloads = DEPS / "cann90-packages"
    downloads.mkdir(parents=True, exist_ok=True)
    prefix = DEPS / "Ascend90"
    env = dict(
        os.environ,
        PATH=str(ROOT / ".venv-npu/bin") + ":" + os.environ["PATH"],
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    for entry in LOCK["cann_packages"]:
        path = downloads / entry["file"]
        if not path.exists():
            partial = path.with_suffix(".partial")
            run(
                "curl",
                "-fL",
                "--retry",
                "3",
                "--connect-timeout",
                "15",
                "-C",
                "-",
                "-H",
                "Referer: https://www.hiascend.com/",
                entry["url"],
                "-o",
                partial,
            )
            partial.rename(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"CANN package verification failed: {path}")
        print(f"Verified {entry['kind']}: {digest}", flush=True)
        command = 'if [ -f "$1/cann/set_env.sh" ]; then source "$1/cann/set_env.sh"; fi\nexec bash "$2" --quiet --install --install-for-all --install-path="$1"'
        run("bash", "-c", command, "umm-cann-install", prefix, path, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("sources", "python", "cann", "host-tools"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("Specify --sources, --python, --cann and/or --host-tools")
    if args.host_tools:
        host_tools()
    if args.sources:
        sources()
    if args.python:
        python()
    if args.cann:
        cann()
