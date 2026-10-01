# Publish the research paper

The paper is `docs/index.html`; GitHub Pages should publish **main → /docs**.

1. Commit and push `docs/`, `scripts/build_capstone_paper.py`,
   `scripts/execute_capstone.py`, and `work/requirements-capstone.txt`.
2. In this repository's **Settings → Pages**, choose **Deploy from a branch**,
   branch **main**, folder **/docs**, then **Save**.
3. Wait for the Pages deployment and check the actual URL shown in that screen.
   The expected URL is `https://mtahar-dev.github.io/flyrank-ml-internship/`.
4. Open the live page and confirm the charts, recommendation disclosure,
   repository/notebook links, and FlyRank credit work.
5. Once verified live, replace `submission/paper_url.txt` with exactly that URL
   on one line. Commit and push the URL file. Never submit a local preview URL.
6. Submit the repository URL: `https://github.com/MTahaR-dev/flyrank-ml-internship`.

The URL file remains a placeholder until deployment is confirmed. This checkpoint
builds the paper; it does not claim the paper is already deployed.

## Regenerate after an analysis change

Run `python scripts/execute_capstone.py`, then
`python scripts/build_capstone_paper.py` from the repository root.
`work/requirements-capstone.txt` records the analysis dependencies. A gated
Hugging Face token or local March parquet is needed for the notebook, but not
for rebuilding the static paper from already-generated aggregate receipts.

## Local preview

From the repo root, run `python -m http.server 8765 --bind 127.0.0.1 --directory docs`.
Open `http://127.0.0.1:8765/`. This local preview is not the deployed URL.

Public assets contain anonymous aggregates and derived recommendations only.
Never add CSV caches, model binaries, tokens, raw exports, page URLs, queries,
client names, or domains to the publication.
