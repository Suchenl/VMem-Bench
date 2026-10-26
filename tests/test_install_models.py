"""vmem-bench-install-models: manifest coverage, layout, dry-run and hash verification (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vmem_bench import install_models as im


def test_manifest_covers_every_scoring_model() -> None:
    man = im.load_manifest()["models"]
    assert {m["name"] for m in im.MODELS} == set(man)
    for m in im.MODELS:
        e = man[m["name"]]
        assert e["target"] == {"root": m["root"], "rel": m["rel"]} and e["files"]
        assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in e["files"])
    assert man["megaloc-weights"]["files"][0]["path"] == "model.safetensors"
    assert {f["path"] for f in man["megaloc-code"]["files"]} >= {"megaloc_model.py"}


def test_layout_matches_embedder_expectations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMSTRATA_WEIGHTS_ROOT", str(tmp_path / "w"))
    monkeypatch.setenv("PUBLIC_MODELS_ROOT", str(tmp_path / "p"))
    by = {m["name"]: im.target_dir(m) for m in im.MODELS}
    assert by["megaloc-code"] == tmp_path / "w/torch_hub/gmberton_MegaLoc_main"
    assert by["megaloc-weights"] == tmp_path / "w/location_embedding/MegaLoc"
    assert by["arcface-buffalo_l"] == tmp_path / "w/human_face_embedding/models/buffalo_l"
    assert by["dinov3-vitb16"] == tmp_path / "p/facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert by["qwen3-vl-32b-instruct"] == tmp_path / "p/Qwen/Qwen3-VL-32B-Instruct"


def test_dry_run_on_empty_roots_lists_fetches(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MEMSTRATA_WEIGHTS_ROOT", str(tmp_path / "w"))
    monkeypatch.setenv("PUBLIC_MODELS_ROOT", str(tmp_path / "p"))
    assert im.main(["install", "--dry-run", "--with-optional"]) == 0
    out = capsys.readouterr().out
    assert out.count("WOULD FETCH") == len(im.MODELS)
    assert not any((tmp_path / "w").rglob("*.safetensors"))


def test_verify_detects_hash_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MEMSTRATA_WEIGHTS_ROOT", str(tmp_path / "w"))
    d = tmp_path / "w/torch_hub/gmberton_MegaLoc_main"
    d.mkdir(parents=True)
    (d / "megaloc_model.py").write_text("x = 1\n")
    good = hashlib.sha256(b"x = 1\n").hexdigest()
    fake = {"models": {"megaloc-code": {"files": [{"path": "megaloc_model.py", "size": 6, "sha256": good}]}}}
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(fake))
    monkeypatch.setattr(im, "MANIFEST", mf)
    assert im.main(["verify", "--only", "megaloc-code"]) == 0
    (d / "megaloc_model.py").write_text("x = 2\n")
    assert im.main(["verify", "--only", "megaloc-code"]) == 1
    assert "sha256 mismatch" in capsys.readouterr().out
