"""Detect overwritten shared files without importing or changing real Triton."""

import base64
import csv
import hashlib
from importlib import metadata
from pathlib import Path

import pytest

from utils import triton_integrity as checker


@pytest.fixture
def installed(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    shared = {name: b"community version" for name in checker.REQUIRED_SHARED}
    ascend = {
        name: b"ascend version"
        for name in checker.REQUIRED_SHARED | checker.REQUIRED_ASCEND
    }
    ascend["triton/backends/ascend/helper.py"] = b"ascend helper"
    # A successful check must locate imports without executing the initializer.
    ascend["triton/__init__.py"] = b"raise RuntimeError('must not import Triton')"
    distributions = {}
    for package, version, payload in [
        ("triton", "3.2.0", shared), ("triton-ascend", "3.2.1", ascend)
    ]:
        info = site / f"{package.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {package}\nVersion: {version}\n")
        rows = []
        for name, data in payload.items():
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append([name, f"sha256={digest}", str(len(data))])
        rows.append(["triton/__pycache__/generated.pyc", "", ""])
        with (info / "RECORD").open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        distributions[package] = metadata.PathDistribution(info)
    monkeypatch.setattr(checker.metadata, "distribution", distributions.__getitem__)
    return site, distributions


def test_ascend_record_wins_for_shared_files_and_bytecode_is_ignored(installed):
    site, _ = installed
    bytecode = site / "triton/__pycache__/generated.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"untracked interpreter output")
    result = checker.verify_triton()
    assert result["shared_files"] == len(checker.REQUIRED_SHARED)
    assert result["checked_files"] == len(checker.REQUIRED_SHARED | checker.REQUIRED_ASCEND) + 1
    assert result["import_path"] == str(site / "triton/__init__.py")


@pytest.mark.parametrize("name, mutation, error", [
    ("triton/runtime/jit.py", "same_size", "SHA-256 mismatch"),
    ("triton/compiler/compiler.py", "truncate", "size mismatch"),
    ("triton/_C/libtriton.so", "delete", "file is missing"),
    ("triton/backends/ascend/driver.py", "delete", "file is missing"),
    ("triton/backends/ascend/helper.py", "same_size", "SHA-256 mismatch"),
    ("triton/backends/ascend/helper.py", "delete", "file is missing"),
])
def test_corruption_is_rejected_even_when_versions_match(installed, name, mutation, error):
    site, _ = installed
    path = site / name
    if mutation == "delete":
        path.unlink()
    elif mutation == "truncate":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(RuntimeError, match=error):
        checker.verify_triton()


@pytest.mark.parametrize("mutation, error", [
    ("missing_record", "RECORD is missing"),
    ("missing_entry", "RECORD is incomplete"),
    ("missing_hash", "lacks SHA-256/size"),
])
def test_incomplete_record_is_rejected(installed, mutation, error):
    _, distributions = installed
    record = Path(distributions["triton-ascend"]._path) / "RECORD"
    if mutation == "missing_record":
        record.unlink()
    else:
        with record.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if mutation == "missing_entry":
            rows = [row for row in rows if row[0] != "triton/runtime/jit.py"]
        else:
            next(row for row in rows if row[0] == "triton/runtime/jit.py")[1] = ""
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    with pytest.raises(RuntimeError, match=error):
        checker.verify_triton()


def test_wrong_version_is_rejected(installed):
    _, distributions = installed
    path = Path(distributions["triton-ascend"]._path) / "METADATA"
    path.write_text("Name: triton-ascend\nVersion: 9.9.9\n")
    with pytest.raises(RuntimeError, match="expected 3.2.1, installed 9.9.9"):
        checker.verify_triton()


def test_missing_distribution_is_reported(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(checker.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="Required distribution is missing"):
        checker.verify_triton()


def test_pythonpath_shadow_is_rejected(installed, tmp_path, monkeypatch):
    shadow = tmp_path / "shadow"
    (shadow / "triton").mkdir(parents=True)
    (shadow / "triton/__init__.py").write_text("# unrelated Triton")
    monkeypatch.syspath_prepend(str(shadow))
    with pytest.raises(RuntimeError, match="import is shadowed"):
        checker.verify_triton()


def test_cli_quiet_success_and_actionable_failure(installed, capsys):
    site, _ = installed
    assert checker.main(["--quiet"]) == 0
    assert capsys.readouterr() == ("", "")
    (site / "triton/_C/libtriton.so").unlink()
    assert checker.main(["--quiet"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "libtriton.so" in captured.err and "setup_mindspeed.py --python" in captured.err
