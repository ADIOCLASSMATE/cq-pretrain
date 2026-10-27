"""Exercise launcher orchestration without allocating accelerators."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest


@pytest.fixture
def environment_project(tmp_path):
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/env_mindspeed.sh"
    shutil.copy2(source, scripts)
    cann = project / ".deps/Ascend90/cann"
    cann.mkdir(parents=True)
    (cann / "set_env.sh").write_text(
        'export UMM_TEST_OPTIONAL="$UMM_TEST_UNSET"\n'
        'export ASCEND_HOME_PATH="$UMM_ENV_ROOT/.deps/Ascend90/cann"\n'
        'return "${UMM_TEST_VENDOR_STATUS:-0}"\n'
    )
    atb = project / ".deps/Ascend90/nnal/atb"
    atb.mkdir(parents=True)
    (atb / "set_env.sh").write_text('export ATB_HOME_PATH="$UMM_ENV_ROOT/.deps/Ascend90/nnal/atb"\n')
    python = project / ".venv-npu/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    utilities = project / "utils"
    utilities.mkdir()
    (utilities / "triton_integrity.py").write_text(
        'import os, sys\n'
        'from pathlib import Path\n'
        'assert sys.argv[1:] == ["--quiet"]\n'
        'assert os.environ["ASCEND_HOME_PATH"] == str(Path(__file__).resolve().parents[1] / ".deps/Ascend90/cann")\n'
        'Path(os.environ["UMM_TEST_CHECK_MARKER"]).touch()\n'
        'sys.exit(int(os.environ.get("UMM_TEST_CHECK_STATUS", 0)))\n'
    )
    return project


def shell_command(shell, code, *args):
    binary = shutil.which(shell)
    if binary is None:
        pytest.skip(f"{shell} is not installed")
    options = ["--noprofile", "--norc"] if shell == "bash" else ["-f"]
    return [binary, *options, "-c", code, "umm-env-test", *map(str, args)]


def environment_for_test(project):
    env = dict(os.environ, UMM_TEST_CHECK_MARKER=str(project / "checked"))
    for key in ["PYTHON_BIN", "UMM_TEST_UNSET", "ASCEND_HOME_PATH", "ATB_HOME_PATH", "ASCEND_WORK_PATH"]:
        env.pop(key, None)
    return env


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_environment_loads_from_each_shell_and_preserves_options(environment_project, shell, strict, relative):
    project = environment_project
    cwd = project if relative else project.parent
    script = "scripts/env_mindspeed.sh" if relative else project / "scripts/env_mindspeed.sh"
    env = environment_for_test(project)
    expected_work = project / ".cache/ascend-work" / (env.get("HOSTNAME") or socket.gethostname())
    if strict:
        # Preserve an explicit interpreter selection as well as the default.
        env["PYTHON_BIN"] = sys.executable
        expected_work = project / "custom ascend work"
        env["ASCEND_WORK_PATH"] = str(expected_work)
    env["UMM_TEST_EXPECTED_WORK"] = str(expected_work)
    code = r'''
if [[ "$3" == 1 ]]; then set -eu; fi
UMM_TEST_PWD="$PWD"
source "$1"
[[ "$UMM_ENV_ROOT" == "$2" && "$PWD" == "$UMM_TEST_PWD" ]] || exit 41
[[ "$ASCEND_HOME_PATH" == "$2/.deps/Ascend90/cann" ]] || exit 42
[[ "$ATB_HOME_PATH" == "$2/.deps/Ascend90/nnal/atb" ]] || exit 43
[[ "$ASCEND_WORK_PATH" == "$UMM_TEST_EXPECTED_WORK" && -d "$ASCEND_WORK_PATH" ]] || exit 44
if [[ "$3" == 1 ]]; then [[ $- == *u* ]]; else [[ $- != *u* ]]; fi
'''
    result = subprocess.run(shell_command(shell, code, script, project, int(strict)),
                            cwd=cwd, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "checked").exists()


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("failure", ["vendor", "integrity"])
def test_environment_propagates_failures_and_restores_nounset(environment_project, shell, failure):
    project = environment_project
    env = environment_for_test(project)
    env["UMM_TEST_VENDOR_STATUS" if failure == "vendor" else "UMM_TEST_CHECK_STATUS"] = "7"
    code = r'''
set -u
if source "$1"; then exit 91; else UMM_TEST_RESULT=$?; fi
[[ $- == *u* ]] || exit 92
exit "$UMM_TEST_RESULT"
'''
    result = subprocess.run(shell_command(shell, code, project / "scripts/env_mindspeed.sh"),
                            cwd=project.parent, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
    assert (project / "checked").exists() == (failure == "integrity")


def test_resume_launcher_initializes_parent_environment_for_post_training_checks(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    validation = tmp_path / "tests/validation"
    validation.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    shutil.copy2(project / "tests/validation/test_mindspeed_resume.sh", validation)
    (scripts / "env_mindspeed.sh").write_text(
        'export UMM_TEST_CANN_READY=1\n'
        'export PYTHON_BIN="${PYTHON_BIN:?test interpreter must be preserved}"\n'
    )
    (validation / "test_train.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        '[[ "${UMM_TEST_CANN_READY:-}" == 1 ]]\n'
        'printf "%s\\n" "$*" >> "$UMM_TEST_TRAIN_CALLS"\n'
        'mkdir -p "$UMM_TEST_OUTPUT/interrupted/checkpoints/checkpoint-2"\n'
        'printf "fixture\\n" > "$UMM_TEST_OUTPUT/interrupted/checkpoints/checkpoint-2/state"\n'
    )
    # The selected interpreter records both parent-level Python stages. The
    # real stdlib snapshot script executes; only the expensive comparator is
    # substituted, and it still requires the inherited CANN environment.
    interpreter = tmp_path / "python-with-environment-check"
    interpreter.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'assert os.environ.get("UMM_TEST_CANN_READY") == "1"\n'
        'with open(os.environ["UMM_TEST_PYTHON_CALLS"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'if sys.argv[1] == "-":\n'
        '    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n'
        'assert sys.argv[1:3] == ["-m", "tests.validation.verify_mcore_run"]\n'
    )
    interpreter.chmod(0o755)
    output = tmp_path / "verification"
    train_calls = tmp_path / "training.txt"
    python_calls = tmp_path / "python.jsonl"
    env = dict(os.environ, PYTHON_BIN=str(interpreter), UMM_TEST_OUTPUT=str(output),
               UMM_TEST_TRAIN_CALLS=str(train_calls), UMM_TEST_PYTHON_CALLS=str(python_calls))
    env.pop("UMM_TEST_CANN_READY", None)
    result = subprocess.run(
        ["bash", str(validation / "test_mindspeed_resume.sh"), str(output)],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    launches = train_calls.read_text().splitlines()
    assert len(launches) == 4 and "--auto-resume" in launches[-1]
    assert [json.loads(row) for row in python_calls.read_text().splitlines()] == [
        ["-", str(output)], ["-m", "tests.validation.verify_mcore_run", str(output)],
    ]
    assert (output / "before_resume_checkpoint/state").read_text() == "fixture\n"
    assert (output / "interrupted/checkpoints/checkpoint-3/partial").is_file()
