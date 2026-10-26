#!/usr/bin/env python3
"""Audit saved manifests and scoring snapshots without generation or scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[3]
T = PAPER / 'tables' / 'experiments'
CELLS = ['frame_text_b3', 'seg_uniform_b3', 'seg_dinokey_b3', 'seg_framererank_b3', 'none_b0', 'last_seen_b3']


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    args = ap.parse_args()
    cohort = json.loads((T / 'trackb_story_cohort.json').read_text())
    coverage = json.loads((T / 'trackb_story_coverage.json').read_text())
    expected = {r['story_id']: r['segments'] for r in coverage['stories']}
    conditions = {}
    problems = []
    for cell in CELLS:
        entries = []
        commits = Counter()
        total = 0
        for story in cohort['stories']:
            run = args.run_root / 'gen' / cell / story / 'name_anchored' / cohort['run_tag'] / 'seed_2026'
            manifest = run / 'run_manifest.json'
            score = args.run_root / 'scores' / (cell+'__s2026') / story / 'score.json'
            if not manifest.exists() or not score.exists():
                problems.append(f'missing manifest or score: {cell}/{story}')
                continue
            d = json.loads(manifest.read_text())
            commits[d['vmem_bench']['commit']] += 1
            if d['seed'] != 2026 or d['segments_ok'] != expected[story] or d['segments_failed']:
                problems.append(f'incomplete or wrong seed: {cell}/{story}')
            if d['vmem_bench']['src_dirty']:
                problems.append(f'dirty generation sources: {cell}/{story}')
            total += d['segments_ok']
            entries.append(dict(story_id=story, manifest_sha256=fingerprint(manifest), score_sha256=fingerprint(score)))
        conditions[cell] = dict(manifests=len(entries), segments_ok=total, commits=dict(commits), inputs=entries)
    sources = ['trackb_story_cohort.json','trackb_story_coverage.json','memory_gain_bootstrap.json',
               'last_seen_bootstrap.json','trackb_judge2.json','trackb_identity_embed.json',
               'trackb_identity_vlm.json','trackb_nolook.json','trackb_quality_control.json',
               'h3/analysis_reference_flux2_wan22.json','h3/analysis_C.json']
    fingerprints = {name: fingerprint(T/name) for name in sources}
    model_manifest = REPO/'src/vmem_bench/scoring/configs/scoring_models_manifest.json'
    report = dict(cohort=cohort, conditions=conditions, frozen_analysis_sha256=fingerprints,
                  scoring_model_manifest_sha256=fingerprint(model_manifest), problems=problems,
                  scope='Read-only metadata/file-fingerprint audit. No model calls, score recalculation, video decoding, pixel regeneration, or remote download verification.',
                  distribution_limit='results/trackB is a local symlink index, not a tracked release bundle. These checks do not prove public download availability.')
    (T/'paper_input_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='inputs'} for k,v in conditions.items()},indent=2))
    print('problems:',problems)
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
