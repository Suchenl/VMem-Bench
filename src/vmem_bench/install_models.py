"""Install / verify every model the Track-B scorers use, in the layout the scorers expect.

Two roots (environment variables, same ones the scorers read):

* ``MEMSTRATA_WEIGHTS_ROOT`` — the scoring weights root (``vmem_bench.scoring.embedder`` default:
  the first ``models/model_weights`` above the package). Holds ArcFace (``human_face_embedding/``)
  and MegaLoc (``torch_hub/gmberton_MegaLoc_main`` + ``location_embedding/MegaLoc/model.safetensors``).
* ``PUBLIC_MODELS_ROOT`` — Hugging Face style ``<org>/<name>`` dirs (default: the weights root).
  Holds DINOv3, GroundingDINO, Qwen3-VL-32B-Instruct (judges) and optionally SigLIP2.

Every file is checked against the SHA256 manifest ``scoring/configs/scoring_models_manifest.json``
(recorded from the maintainers' local copies). Files already present with the right hash are skipped.

    vmem-bench-install-models install --dry-run      # what would be fetched, and where
    vmem-bench-install-models install [--only dinov3-vitb16,...] [--with-optional]
    vmem-bench-install-models verify                  # hash every local file against the manifest
    vmem-bench-install-models manifest --write        # maintainers: record hashes of local copies

Downloads use ``huggingface_hub`` (HF repos; DINOv3 is gated: accept its license and set ``HF_TOKEN``)
and plain HTTPS for the InsightFace pack and the MegaLoc code. Nothing is downloaded with --dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "scoring" / "configs" / "scoring_models_manifest.json"

MODELS = [
    {"name": "dinov3-vitb16", "root": "public", "rel": "facebook/dinov3-vitb16-pretrain-lvd1689m",
     "source": {"type": "hf", "repo": "facebook/dinov3-vitb16-pretrain-lvd1689m"},
     "used_by": ["embedder dinov3 (Track A redundancy_sim, Track B trackB-embed-1.0 crops/frames)"]},
    {"name": "grounding-dino-base", "root": "public", "rel": "IDEA-Research/grounding-dino-base",
     "source": {"type": "hf", "repo": "IDEA-Research/grounding-dino-base"},
     "used_by": ["trackB-embed-1.0 entity localization"]},
    {"name": "arcface-buffalo_l", "root": "weights", "rel": "human_face_embedding/models/buffalo_l",
     "source": {"type": "zip", "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                "strip": ""},
     "used_by": ["embedder arcface (trackB-embed-1.0 faces)"]},
    {"name": "megaloc-code", "root": "weights", "rel": "torch_hub/gmberton_MegaLoc_main",
     "source": {"type": "zip", "url": "https://github.com/gmberton/MegaLoc/archive/refs/heads/main.zip",
                "strip": "MegaLoc-main/"},
     "used_by": ["embedder megaloc (model code)"]},
    {"name": "megaloc-weights", "root": "weights", "rel": "location_embedding/MegaLoc",
     "source": {"type": "hf", "repo": "gberton/MegaLoc"},
     "used_by": ["embedder megaloc (trackB-embed-1.0 locations)"]},
    {"name": "qwen3-vl-32b-instruct", "root": "public", "rel": "Qwen/Qwen3-VL-32B-Instruct",
     "source": {"type": "hf", "repo": "Qwen/Qwen3-VL-32B-Instruct"},
     "used_by": ["judge: trackB-end2end-1.0, trackB-identity-1.0, visual-coverage (vLLM served as qwen3-vl-32b)"]},
    {"name": "siglip2-base", "root": "public", "rel": "google/siglip2-base-patch16-224", "optional": True,
     "source": {"type": "hf", "repo": "google/siglip2-base-patch16-224"},
     "used_by": ["embedder siglip2 (optional secondary axis)"]},
]

KEEP_SUFFIXES = (".safetensors", ".json", ".txt", ".onnx", ".py", ".model", ".bin")
SKIP_NAMES = {"configuration.json"}  # ModelScope metadata, not part of the HF repos


def weights_root() -> Path:
    env = os.environ.get("MEMSTRATA_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "models" / "model_weights"
        if cand.is_dir():
            return cand
    return here.parents[2] / "models" / "model_weights"


def public_root() -> Path:
    env = os.environ.get("PUBLIC_MODELS_ROOT")
    return Path(env).expanduser().resolve() if env else weights_root()


def target_dir(m: dict) -> Path:
    return (public_root() if m["root"] == "public" else weights_root()) / m["rel"]


def local_candidates(m: dict) -> list[Path]:
    """Where an existing local copy may live (target first, then HF-cache snapshots)."""
    out = [target_dir(m)]
    if m["source"]["type"] == "hf":
        repo = m["source"]["repo"]
        for base in (public_root() / m["rel"], weights_root() / "hub" / f"models--{repo.replace('/', '--')}"):
            out += sorted(p for p in (base / "snapshots").glob("*") if p.is_dir())
    return out


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _model_files(d: Path) -> list[Path]:
    files = []
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if any(part.startswith((".", "__pycache__")) for part in rel.parts) or p.name in SKIP_NAMES:
            continue
        if p.is_file() and p.suffix in KEEP_SUFFIXES:
            files.append(p)
    return files


def _portable(d: Path) -> str:
    """Record a local copy's location relative to the root variables (no personal paths in git)."""
    for var, root in (("MEMSTRATA_WEIGHTS_ROOT", weights_root()), ("PUBLIC_MODELS_ROOT", public_root())):
        try:
            return "${%s}/%s" % (var, Path(d).resolve().relative_to(root))
        except ValueError:
            continue
    return Path(d).name


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {"models": {}}


def find_local(m: dict, entry: dict | None) -> Path | None:
    for d in local_candidates(m):
        if not d.is_dir() or (d / "snapshots").is_dir():  # an HF cache root, not a model dir
            continue
        if entry is None:
            if _model_files(d):
                return d
        elif all((d / f["path"]).is_file() for f in entry["files"]):
            return d
    return None


def cmd_manifest(a: argparse.Namespace) -> int:
    out = {"_doc": "SHA256 of every scoring-model file, recorded from local copies; see vmem_bench.install_models",
           "models": {}}
    for m in MODELS:
        d = find_local(m, None)
        if d is None:
            print(f"[{m['name']}] no local copy found; not recorded", file=sys.stderr)
            continue
        files = _model_files(d)
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            hashes = list(ex.map(sha256_file, files))
        out["models"][m["name"]] = {
            "target": {"root": m["root"], "rel": m["rel"]}, "source": m["source"], "optional": bool(m.get("optional")),
            "used_by": m["used_by"], "recorded_from": _portable(d),
            "files": [{"path": str(f.relative_to(d)), "size": f.stat().st_size, "sha256": h} for f, h in zip(files, hashes)],
        }
        print(f"[{m['name']}] {len(files)} files from {d}")
    text = json.dumps(out, indent=1) + "\n"
    if a.write:
        MANIFEST.write_text(text)
        print(f"wrote {MANIFEST}")
    else:
        print(text)
    return 0


def _check(d: Path, entry: dict, jobs: int) -> list[str]:
    def one(f: dict) -> str | None:
        p = d / f["path"]
        if not p.is_file():
            return f"missing {p}"
        if p.stat().st_size != f["size"]:
            return f"size mismatch {p}"
        return None if sha256_file(p) == f["sha256"] else f"sha256 mismatch {p}"

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return [r for r in ex.map(one, entry["files"]) if r]


def cmd_verify(a: argparse.Namespace) -> int:
    man = load_manifest()["models"]
    bad = 0
    for m in MODELS:
        if a.only and m["name"] not in a.only:
            continue
        entry = man.get(m["name"])
        if entry is None:
            print(f"[{m['name']}] not in manifest")
            continue
        d = find_local(m, entry)
        if d is None:
            print(f"[{m['name']}] MISSING (expected at {target_dir(m)})")
            bad += 0 if m.get("optional") else 1
            continue
        errs = _check(d, entry, a.jobs)
        size = sum(f["size"] for f in entry["files"])
        print(f"[{m['name']}] {'OK' if not errs else 'FAIL'} {len(entry['files'])} files {size / 1e9:.2f} GB at {d}"
              + ("" if d == target_dir(m) else f" (scorers resolve it; canonical target {target_dir(m)})"))
        for e in errs:
            print("   ", e)
        bad += bool(errs)
    return 1 if bad else 0


def _download(m: dict, entry: dict, dest: Path) -> None:
    src = m["source"]
    dest.mkdir(parents=True, exist_ok=True)
    if src["type"] == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=src["repo"], local_dir=str(dest), allow_patterns=[f["path"] for f in entry["files"]])
        return
    with urllib.request.urlopen(src["url"], timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(src.get("strip", "")):
                continue
            rel = name[len(src.get("strip", "")):]
            base = Path(rel).name if src.get("strip", "") == "" else rel
            out = dest / base
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def cmd_install(a: argparse.Namespace) -> int:
    man = load_manifest()["models"]
    rc = 0
    for m in MODELS:
        if a.only and m["name"] not in a.only:
            continue
        if m.get("optional") and not a.with_optional and not a.only:
            print(f"[{m['name']}] optional; skipped (use --with-optional)")
            continue
        entry = man[m["name"]]
        dest = target_dir(m)
        have = find_local(m, entry)
        size = sum(f["size"] for f in entry["files"])
        if have is not None and (a.dry_run or not _check(have, entry, a.jobs)):
            print(f"[{m['name']}] present at {have}; skip" + (" (hash check: run `verify`)" if a.dry_run else ""))
            continue
        src = m["source"]
        what = src.get("repo") or src.get("url")
        print(f"[{m['name']}] {'WOULD FETCH' if a.dry_run else 'fetch'} {len(entry['files'])} files "
              f"({size / 1e9:.2f} GB) from {src['type']}:{what} -> {dest}")
        if a.dry_run:
            continue
        _download(m, entry, dest)
        errs = _check(dest, entry, a.jobs)
        print(f"[{m['name']}] {'verified' if not errs else 'VERIFY FAILED'}")
        for e in errs:
            print("   ", e)
        rc |= bool(errs)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("install", cmd_install), ("verify", cmd_verify), ("manifest", cmd_manifest)):
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=8, help="parallel hash workers")
        p.set_defaults(fn=fn)
        if name in ("install", "verify"):
            p.add_argument("--only", default="", help="comma list of model names")
        if name == "install":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--with-optional", action="store_true")
        if name == "manifest":
            p.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(a, "only"):
        a.only = [x for x in a.only.split(",") if x]
    print(f"weights root: {weights_root()}\npublic models root: {public_root()}")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
