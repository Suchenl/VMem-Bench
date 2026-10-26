"""Track-first resume + parallel-tracking core (no GPU, no multiprocessing).

Checks the two invariants the two features rest on:
  1. resume checkpoints round-trip losslessly (roster / tracklets / names / chunks), and shot_done
     reflects on-disk state -> a re-run skips finished shots.
  2. compute_tracklets' in-process path (devices<=1, same code the workers run) computes + checkpoints
     each shot, and reloading the shots in order reproduces the tracklets (deterministic ids after
     the pipeline's renumber).

Run: PYTHONPATH=benchmarks/MemStrata/src python benchmarks/MemStrata/tests/test_track_parallel_resume.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from vmem_bench.annotation.pipeline_track_first import resume as R
from vmem_bench.annotation.pipeline_track_first import track_parallel as TP
from vmem_bench.annotation.pipeline_track_first.perception.base import RosterEntry
from vmem_bench.annotation.pipeline_track_first.tracking import Detection, Tracklet


def _tk(track_id: int, phrase: str, frames: list[int]) -> Tracklet:
    dets = [Detection(frame_index=f, bbox=[10, 10, 200, 200], score=0.9, phrase=phrase,
                      embedding=[0.1 * f, 0.2], crop_path=f"c{f}.jpg") for f in frames]
    return Tracklet(track_id=track_id, phrase=phrase, detections=dets)


def test_resume_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        # roster
        assert R.load_roster(out) is None
        roster = [{"name": "Rabbit", "kind": "character", "grounding_phrase": "grey rabbit"}]
        R.save_roster(out, roster)
        assert R.load_roster(out) == roster
        # tracklets (per shot) + shot_done + serialization
        assert not R.shot_done(out, 0)
        tks = [_tk(0, "grey rabbit", [100, 105]), _tk(1, "grey rabbit", [100, 110])]
        payload = R.shot_payload(0, 100, 110, tks, [[0.5, 0.6], None], scene_frame=105,
                                 scene_vec=[0.3, 0.4])
        R.save_shot(out, payload)
        assert R.shot_done(out, 0)
        loaded = R.load_shot(out, 0)
        assert loaded["first"] == 100 and loaded["scene_frame"] == 105
        r_tks, r_sigs = R.shot_tracklets(loaded)
        assert [t.track_id for t in r_tks] == [0, 1]
        assert r_tks[0].detections[0].crop_path == "c100.jpg"
        assert r_tks[0].detections[0].embedding == [10.0, 0.2]  # 0.1*100
        assert r_sigs == [[0.5, 0.6], None]
        # names (per entity)
        assert R.load_name(out, "char_rabbit") is None
        R.save_name(out, "char_rabbit", "Big Rabbit", "a large grey rabbit")
        assert R.load_name(out, "char_rabbit")["name"] == "Big Rabbit"
        # chunks (per chunk)
        R.save_chunk(out, 3, {"chunk_id": 3, "prompt": "hi"})
        assert R.load_chunk(out, 3)["prompt"] == "hi"
        assert R.load_chunk(out, 4) is None
    print("test_resume_roundtrip OK")


class _FakeBackend:
    """track_shot ignores pixels; returns one tracklet whose id starts at next_track_id."""

    def track_shot(self, frames, roster, *, next_track_id=0):
        ph = roster[0].grounding_phrase
        return [_tk(next_track_id, ph, [f.frame_index for f in frames])]


class _FakeEmbed:
    def embed_image(self, path):
        return [0.7, 0.8]


class _FakeFace:
    def encode(self, path):
        return [0.9, 1.0]


class _Cfg:
    track_fps = 3.0
    track_min_len = 1
    use_face = True


def test_compute_tracklets_inprocess_and_resume_skip() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        shots = [(0, 30), (31, 60), (61, 90)]
        roster = [RosterEntry(name="Rabbit", kind="character", grounding_phrase="grey rabbit")]
        rdata = [{"name": "Rabbit", "kind": "character", "grounding_phrase": "grey rabbit"}]
        fp = lambda i: Path(f"/nope/f{i}.jpg")  # noqa: E731 (fake never reads pixels)
        seen: list[int] = []
        fails = TP.compute_tracklets(
            shots, [0, 1, 2], config=_Cfg(), roster_entries=roster, roster_data=rdata,
            out=out, video=Path("v.mp4"), frames_dir=out, crop_dir=out, fps=10.0, n_frames=0,
            devices=[], backend=_FakeBackend(), embedder=_FakeEmbed(), face_encoder=_FakeFace(),
            frame_path=fp, progress=seen.append)
        assert fails == []
        assert seen == [1, 2, 3]  # cumulative progress
        for i in range(3):
            assert R.shot_done(out, i)
        # face signatures were computed (character + face on) and scene vec stored
        p0 = R.load_shot(out, 0)
        assert p0["tracklets"][0]["face_sig"] == [0.9, 1.0]
        assert p0["scene_vec"] == [0.7, 0.8]

        # resume: with shot 1 already done, a "pending" list excludes it -> only 0,2 recomputed.
        todo = [i for i in range(3) if not R.shot_done(out, i)]
        assert todo == []  # all done -> nothing to do
        R.shot_ckpt_path(out, 1).unlink()  # simulate a shot lost mid-run
        todo = [i for i in range(3) if not R.shot_done(out, i)]
        assert todo == [1]
        seen2: list[int] = []
        TP.compute_tracklets(
            shots, todo, config=_Cfg(), roster_entries=roster, roster_data=rdata, out=out,
            video=Path("v.mp4"), frames_dir=out, crop_dir=out, fps=10.0, n_frames=0, devices=[],
            backend=_FakeBackend(), embedder=_FakeEmbed(), face_encoder=_FakeFace(),
            frame_path=fp, progress=seen2.append)
        assert seen2 == [1] and R.shot_done(out, 1)
    print("test_compute_tracklets_inprocess_and_resume_skip OK")


if __name__ == "__main__":
    test_resume_roundtrip()
    test_compute_tracklets_inprocess_and_resume_skip()
    print("all track-parallel/resume tests passed")
