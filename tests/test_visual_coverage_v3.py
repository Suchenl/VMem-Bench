"""Contract guards for refs-only per-reference visual-coverage-3.0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vmem_bench.scoring import judge_service, visual_coverage


def _image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (12, 8), color).save(path)
    return str(path)


class _FakeJudge:
    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers
        self.calls: list[tuple[list[dict], dict]] = []

    def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        image_url = content[1]["image_url"]["url"]
        return json.dumps({"entity_ids": self.answers[image_url]})


def _roster() -> list[dict[str, str]]:
    return [
        {
            "entity_id": "char_001",
            "name": "person",
            "kind": "character",
            "description": "a person",
        },
        {
            "entity_id": "loc_001",
            "name": "room",
            "kind": "location",
            "description": "a room",
        },
    ]


def test_call_judge_http_forwards_structured_response_format(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"entity_ids\\":[]}"}}]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(judge_service.urllib.request, "urlopen", fake_urlopen)
    response_format = {"type": "json_schema", "json_schema": {"name": "x"}}
    raw = judge_service.call_judge_http(
        "http://judge/v1",
        "model",
        [{"type": "text", "text": "prompt"}],
        response_format=response_format,
    )
    assert raw == '{"entity_ids":[]}'
    assert captured["payload"]["response_format"] == response_format
    assert captured["timeout"] == 900


@pytest.mark.parametrize(
    "raw",
    [
        '{"entity_ids":["char_001"]}`',
        '{"entity_ids":["char_001","char_001"]}',
        '{"entity_ids":["outside_roster"]}',
        '{"entity_ids":["char_001"],"missing":[]}',
    ],
)
def test_v3_parser_rejects_non_strict_or_invalid_payloads(raw):
    with pytest.raises((json.JSONDecodeError, ValueError)):
        visual_coverage._parse_v3_reference(
            raw,
            {"char_001", "loc_001"},
        )


def test_v3_per_ref_multilabel_is_ordered_and_refs_only(tmp_path):
    refs = [
        _image(tmp_path / "red.png", (255, 0, 0)),
        _image(tmp_path / "blue.png", (0, 0, 255)),
        _image(tmp_path / "green.png", (0, 255, 0)),
    ]
    answers = {
        visual_coverage._img(refs[0])["image_url"]["url"]: [
            "char_001",
            "loc_001",
        ],
        visual_coverage._img(refs[1])["image_url"]["url"]: [],
        visual_coverage._img(refs[2])["image_url"]["url"]: ["loc_001"],
    }
    judge = _FakeJudge(answers)
    score, detail = visual_coverage.score_segment(
        7,
        refs,
        ["char_001", "loc_001"],
        ["char_001", "loc_001"],
        "legacy roster text",
        "SECRET SEGMENT PROMPT",
        Path("/must/not/be/sent.mp4"),
        judge,
        "qwen3-vl-32b",
        metric_version=visual_coverage.PER_REF_METRIC_VERSION,
        roster=_roster(),
        ref_workers=3,
    )

    assert len(judge.calls) == 3
    for content, kwargs in judge.calls:
        assert [row["type"] for row in content] == ["text", "image_url"]
        assert "SECRET SEGMENT PROMPT" not in content[0]["text"]
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 256
        schema = kwargs["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
    assert [row["pred_entities"] for row in detail["refs"]] == [
        ["char_001", "loc_001"],
        [],
        ["loc_001"],
    ]
    assert score.precision == 0.6667
    assert score.recall == 1.0
    assert score.f1 == 0.8
    assert detail["missing_pred"] == []
    assert detail["judge_requests"] == 3
    assert score.redundancy_vlm is None
    assert score.selection_efficiency is None


def test_v3_cache_reuses_only_validated_exact_payloads(tmp_path):
    ref = _image(tmp_path / "ref.png", (255, 0, 0))
    image_url = visual_coverage._img(ref)["image_url"]["url"]
    judge = _FakeJudge({image_url: ["char_001", "loc_001"]})
    cache = visual_coverage._V3JudgeCache(tmp_path / "cache")
    kwargs = {
        "cid": 1,
        "refs": [ref],
        "present": ["char_001", "loc_001"],
        "continuity": ["char_001", "loc_001"],
        "roster_txt": "unused",
        "prompt": "unused",
        "clip": None,
        "api": judge,
        "model": "qwen3-vl-32b",
        "metric_version": visual_coverage.PER_REF_METRIC_VERSION,
        "roster": _roster(),
        "judge_cache": cache,
    }
    first_score, first = visual_coverage.score_segment(**kwargs)
    second_score, second = visual_coverage.score_segment(**kwargs)

    assert first_score == second_score
    assert len(judge.calls) == 1
    assert first["judge_requests"] == 1
    assert first["judge_cache_hits"] == 0
    assert second["judge_requests"] == 0
    assert second["judge_cache_hits"] == 1
    assert first["refs"][0]["cache_key"] == second["refs"][0]["cache_key"]


def test_v3_run_skips_target_clip_and_records_contract(
    tmp_path,
    monkeypatch,
):
    movie = tmp_path / "assets" / "Dataset" / "Movie"
    gold = movie / "gold"
    gold.mkdir(parents=True)
    (gold / "chunk_annotations.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": 0,
                        "present": ["char_001"],
                        "first_appearances": [],
                        "prompt": "must not reach judge",
                        "seconds_span": [0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (gold / "entity_registry.json").write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "char_001",
                        "name": "person",
                        "kind": "character",
                        "description": "a person",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ref = _image(tmp_path / "ref.png", (255, 0, 0))
    output_root = tmp_path / "tracka"
    system = "fake__B16"
    selection = (
        output_root
        / system
        / "Dataset"
        / "Movie"
        / "visual_selections"
        / f"{system}.json"
    )
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": 0,
                        "selected": [
                            {
                                "representations": [
                                    {"crop_abspath": ref}
                                ]
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    image_url = visual_coverage._img(ref)["image_url"]["url"]
    judge = _FakeJudge({image_url: ["char_001"]})
    monkeypatch.setattr(visual_coverage, "_TRACKA_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        visual_coverage,
        "_cut_clip",
        lambda *_args, **_kwargs: pytest.fail("v3 must not cut a target clip"),
    )
    monkeypatch.setattr(
        visual_coverage,
        "_get_embedder",
        lambda: pytest.fail("v3 must not load the redundancy embedder"),
    )
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not read by v3")

    summary = visual_coverage.run(
        movie,
        system,
        video,
        tmp_path / "scores",
        judge,
        "qwen3-vl-32b",
        "ffmpeg",
        workers=1,
        metric_version=visual_coverage.PER_REF_METRIC_VERSION,
    )

    assert summary["metric_version"] == "visual-coverage-3.0"
    assert summary["judge_contract"] == (
        visual_coverage.V3_CONTRACT_VERSION
    )
    assert summary["judge_requests"] == 1
    assert summary["judge_cache_hits"] == 0
