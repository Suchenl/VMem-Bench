# How to obtain source videos

VMem-Bench **does not ship pixels**. Hugging Face `Suchenl/VMem-Bench` is gold JSON and Track B prompts only. Track A still needs the original films on disk so the runner can cut real segments.

Set one root (default is this repo's `data/`):

```bash
export VMEM_DATASETS_ROOT="$PWD/data"   # or any writable directory
```

`assets/trackA/dataset_dirs.txt` expands that variable:

```
BlenderOpenMovies: ${VMEM_DATASETS_ROOT}/BlenderOpenMovies/Videos
LSMDC:             ${VMEM_DATASETS_ROOT}/LSMDC/LSMDC_Videos_Stitched
```

Check what you already have (stats expected paths only; does not walk the tree):

```bash
python scripts/check_source_videos.py
```

Track B scoring uses authored prompts / GT JSON. **No source films.**

---

## 1. Smoke path (one film, no application)

Big Buck Bunny 720p, official Blender Foundation / Peach file (CC BY):

```bash
bash scripts/prepare_blender.sh
```

This downloads

`https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov`

into

`$VMEM_DATASETS_ROOT/BlenderOpenMovies/Videos/big_buck_bunny/big_buck_bunny_720p_h264.mov`

Gold for this id is already in `assets/trackA/BlenderOpenMovies/big_buck_bunny/`. Then:

```bash
python scripts/doctor.py
bash scripts/run_tracka_smoke.sh   # needs sibling MemStrata + PUBLIC_MODELS_ROOT + GPU perception
```

Override the URL with `BBB_URL=...` if the Peach CDN is blocked.

---

## 2. Full Blender / CC corpus (Track A)

Gold directories live under `assets/trackA/BlenderOpenMovies/<movie_id>/`. Put **one** full-length video file here:

```
$VMEM_DATASETS_ROOT/BlenderOpenMovies/Videos/<movie_id>/<anything>.{mp4,mov,mkv,webm,avi}
```

The runner takes the first video file in that folder. Use the **full film**, not a trailer. Keep each film's original license (usually CC-BY or CC-BY-SA) next to your copy.

| `movie_id` (must match gold) | Where to get the film |
|---|---|
| `big_buck_bunny` | `scripts/prepare_blender.sh` or [Peach / BBB](https://peach.blender.org/) |
| `elephants_dream` | [Elephants Dream](https://orange.blender.org/) · [download.blender.org/ED](https://download.blender.org/ED/) |
| `sintel` | [Sintel](https://durian.blender.org/) · [download.blender.org/durian](https://download.blender.org/durian/movies/) |
| `tears_of_steel` | [Tears of Steel](https://mango.blender.org/) · [download.blender.org/mango](https://download.blender.org/mango/) |
| `cosmos_laundromat_first_cycle` | [Gooseberry / Cosmos Laundromat](https://gooseberry.blender.org/) |
| `spring` | [Blender Studio · Spring](https://studio.blender.org/films/spring/) |
| `sprite_fright` | [Blender Studio · Sprite Fright](https://studio.blender.org/films/sprite-fright/) |
| `caminandes_1_llama_drama`, `caminandes_2_gran_dillama`, `caminandes_3_llamigos` | [Caminandes](https://studio.blender.org/films/caminandes/) |
| `charge`, `glass_half`, `hero`, `wing_it`, `daily_dweebs`, `singularity` | [Blender Studio films](https://studio.blender.org/films/) |
| `pepper_carrot_ep3`, `pepper_carrot_ep6` | [Pepper & Carrot](https://www.peppercarrot.com/) / Blender Studio |
| `morevna_ep3`, `morevna_ep4`, `morevna_underground` | [Morevna Project](https://morevnaproject.org/) (CC; not a Blender Foundation Open Movie) |
| `sita_sings_the_blues_part1`, `sita_sings_the_blues_part2` | [Sita Sings the Blues](https://www.sitasingstheblues.com/) (Nina Paley, CC-BY-SA; split to match our two gold ids) |

Index of Open Movies: https://studio.blender.org/films/

Do **not** vendor these files in git or on Hugging Face.

---

## 3. LSMDC (Track A, gated)

We cannot host Hollywood pixels. You must obtain them from LSMDC / MPII-MD.

### Apply

1. Open the official download page: https://sites.google.com/site/describingmovies/download
2. Request access (the page routes you through the MPII Movie Description / LSMDC application). They issue credentials; there is **no** public CDN.
3. Download the **video files** for the titles whose ids appear in `assets/trackA/LSMDC/` (or the full gold on Hugging Face `trackA/LSMDC/`).
4. You do **not** need LSMDC's official AD / csv annotations. Scoring uses our gold JSON (CC BY 4.0). Do not redistribute LSMDC's official csv/AD either.

If the MPII download portal is being rebuilt, use the status note on https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset and the same Google Sites page.

### Layout after you have the files

LSMDC is often distributed as **many clips per title**. Concatenate clips of one movie in chronological order into a **single** file:

```
$VMEM_DATASETS_ROOT/LSMDC/LSMDC_Videos_Stitched/<movie_id>.mp4
```

Example (smoke title used in adapter docs):

```
$VMEM_DATASETS_ROOT/LSMDC/LSMDC_Videos_Stitched/0001_American_Beauty.mp4
```

`<movie_id>` must equal the gold directory name (including the numeric prefix). Other extensions (`.mkv`, `.mov`, …) are accepted. Flat file next to the stitched root, **not** a per-movie subdirectory.

ffmpeg concat example once you have ordered clips:

```bash
# files.txt: one line per clip, in time order
#   file 'clip_001.mp4'
#   file 'clip_002.mp4'
ffmpeg -f concat -safe 0 -i files.txt -c copy \
  "$VMEM_DATASETS_ROOT/LSMDC/LSMDC_Videos_Stitched/0001_American_Beauty.mp4"
```

### Cite

When you use LSMDC titles, cite:

> Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, Bernt Schiele. Movie Description. *International Journal of Computer Vision*, 2017.

### Movie ids in the public gold (git sample + HF full set)

The ids are the LSMDC directory names under `assets/trackA/LSMDC/` and `trackA/LSMDC/` on Hugging Face. `python scripts/check_source_videos.py` prints every missing id.

---

## 4. Gold JSON (not videos)

```bash
huggingface-cli download Suchenl/VMem-Bench --repo-type dataset --local-dir ./VMem-Bench-data
```

Git already contains small samples under `assets/trackA/` for tests. Full Track A gold is on Hugging Face.

---

## 5. Annotate a new video

You can use the same repository to create a new annotation package; this is
separate from downloading the frozen gold above. The input must be one
time-continuous full-length video (not a directory of unrelated clips).

The pipeline has two modes:

- **Diagnostic proposal:** automatic roster discovery is allowed, but the
  output is not production gold and cannot be frozen.
- **Production gold:** provide a human-confirmed canonical roster with
  `ROSTER_SEED`; review and freeze all remaining blockers before publishing.

The pipeline needs OpenAI-compatible VLM endpoints. Start at least one
annotator/reviewer pair (8B is the lightweight default; use a larger judge
where available):

```bash
export PUBLIC_MODELS_ROOT="$HOME/public_models"
export VLLM_ENV=/path/to/your/vllm-env
MODEL_SIZE=8B bash scripts/get_trackA_assets/servers/start_annotation_vllm.sh 0 8001
MODEL_SIZE=8B bash scripts/get_trackA_assets/servers/start_annotation_vllm.sh 1 8002
```

Then run the resumable S1–S7 pipeline:

```bash
export PY=/path/to/your/vllm-env/bin/python
export VIDEO=/path/to/movie.mp4
export MOVIE_ID=my_movie
export OUT="$PWD/annotation_runs/$MOVIE_ID"

# Diagnostic proposal (cannot be frozen):
PROPOSAL_ONLY=1 CLIENT_GPU=2 PY="$PY" \
  bash scripts/get_trackA_assets/core/run_annotation.sh

# Production annotation (requires a human-confirmed roster JSON):
ROSTER_SEED=/path/to/human_confirmed_roster.json CLIENT_GPU=2 \
  PY="$PY" bash scripts/get_trackA_assets/core/run_annotation.sh
```

The output is written below `$OUT`: stage artifacts are resumable, while
`gold/` is accepted only after the human review/freeze gates pass. Inspect the
full CLI and all available options without starting a service:

```bash
PYTHONPATH=src "$PY" -m vmem_bench.annotation.pipeline_track_first.run --help
```

The launcher fails early with an actionable message when `PUBLIC_MODELS_ROOT`,
the source video, the roster (production mode), or an endpoint is missing.
Model weights and the VLM serving environment are not bundled; keep
third-party video/model licenses.

---

## 6. What we will not give you

| Artifact | Why |
|---|---|
| Blender / LSMDC video bytes | Third-party copyright / LSMDC access terms |
| LSMDC official csv / audio description | Same gated release as the videos |
| Model weights | Download yourself; see MemStrata `MODELS.md` |
| A hash file for every LSMDC title | We do not republish LSMDC payloads; verify against **your** official download |
