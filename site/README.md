# VMem-Bench project page

Static project page for https://suchenl.github.io/VMem-Bench/ . No build tool,
frontend framework, CDN, model service or third-party analytics is required.
Open `index.html` locally, or serve this directory with a static HTTP server.

## Updating the page

Edit `index.html`, `assets/site.css` and `assets/site.js` here in the development
repository. The following maintainer helpers live in the private development
checkout; run them there before exporting, not inside the public checkout. Refresh generated content from the canonical paper assets:

```bash
python3 assets/_paper/VMem-Bench/cvpr_27_submission/scripts/export_arxiv.py
python3 scripts/packaging/build_project_page.py
```

The helper copies only generated Track-B stills, the public arXiv PDF, and measured
numbers from the existing saved analysis snapshots. No source-film footage is
used for the interactive examples. `assets/data.js` records the source hashes;
do not hand-edit measured values or relabel them as new experiments.

The figures are from earlier rollouts, not the main quantitative rerun. The page
states that distinction. Its quantitative comparison shows four retrieval policies
against no memory; Last-seen remains a separately described diagnostic control.

## GitHub Pages

The curated public export includes `site/` and `.github/workflows/pages.yml`.
After pushing the public repository, choose **Settings → Pages → Source → GitHub
Actions**. The workflow deploys `site/` on pushes to `main` that change the site,
or can be started manually. No custom domain is configured. The intended URL is
not a claim of an already-live deployment.

This public page and its resource links must never be inserted into anonymous
CVPR submission sources. The arXiv exporter adds resource links only to its
separate generated abstract.

## Design sources

- Author-owned AgentHarness3D / ManyWorlds project pages: reset/container base,
  responsive sections and evidence-led media presentation.
- OpenResearchAssets `examples/sample-project.html`: question-first hero,
  responsive two-column evidence structure and resource/limitations separation.
- SuperMimic (https://supermimic.github.io/): visual inspiration for typography,
  whitespace, media-led examples and section pacing; no SuperMimic media copied.

The template sites contain paper-specific data and interface code that are not
applicable here. Only the small reusable layout foundation was adapted; VMem-Bench
content and interactive metric rendering are specific to this study.
