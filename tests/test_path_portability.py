"""Static guards for portable benchmark paths."""

from __future__ import annotations

import subprocess
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {".py", ".sh", ".toml", ".json", ".yaml", ".yml"}
FORBIDDEN_PATHS = (
    "/m2v_intern/" + "chenyuzhuo03/",
    "/home/" + "chenyuzhuo03/",
    "/" + "root/",
)


def _tracked_source_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "scripts", "tests"],
        cwd=BENCH_ROOT,
        check=True,
        capture_output=True,
    )
    files = [
        BENCH_ROOT / relative.decode("utf-8")
        for relative in result.stdout.split(b"\0")
        if relative and Path(relative.decode("utf-8")).suffix in SCANNED_SUFFIXES
    ]
    files = [path for path in files if path.is_file()]
    assert files, "git returned no tracked benchmark source files"
    return files


def test_tracked_sources_have_no_personal_absolute_paths() -> None:
    violations: list[str] = []
    for path in _tracked_source_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text:
                violations.append(f"{path.relative_to(BENCH_ROOT)}: {forbidden}")
    assert not violations, "personal absolute paths remain:\n" + "\n".join(violations)


def test_tracked_shell_scripts_parse() -> None:
    scripts = [path for path in _tracked_source_files() if path.suffix == ".sh"]
    result = subprocess.run(
        ["bash", "-n", *map(str, scripts)],
        cwd=BENCH_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
