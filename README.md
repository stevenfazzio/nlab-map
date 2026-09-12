# nlab-map

An interactive datamap of the [nLab](https://ncatlab.org) wiki: every concept page embedded, laid out
in two dimensions, and named by region. Live at **[stevenfazzio.com/nlab-map](https://stevenfazzio.com/nlab-map/)**.

## What is on the map

12,201 pages from a 2026-09-12 snapshot of the wiki: concept pages plus the wiki's "reference" pages
(entries for specific books, papers, and conferences). People, navigation templates, disambiguation
pages, page fragments, and pages with no self-description are left out.

Each page is embedded from its title plus the section where it describes itself (the Idea section, else
the Definition, else the Statement, else the preamble before the first heading), capped at 400 words,
with [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B). UMAP lays the embeddings
out in 2-d, [Toponymy](https://github.com/TutteInstitute/toponymy) clusters that layout and names the
regions with Claude Haiku, and [DataMapPlot](https://github.com/TutteInstitute/datamapplot) renders it.

Hover shows the page's summary, its primary Context sidebar heading, which section the embedding came
from, its inbound-link count, and its last-revised date. Search covers titles, redirect names, summaries,
and Context headings. Click opens the page on ncatlab.org. Marker size is log inbound links. Colormaps:
page structure, primary Context heading, date last revised.

## Pipeline

Numbered stages under `pipeline/`, each reading and writing files under `data/` (git-ignored). The
rendered site is committed under `docs/` and served by GitHub Pages.

| Stage | Reads | Writes |
|---|---|---|
| `01_fetch.py` | GitHub mirrors `ncatlab/nlab-content` (markdown) and `ncatlab/nlab-content-html` | `data/raw/`, `data/raw/mirror_commits.json` |
| `02_build_corpus.py` | `data/raw/` | `data/corpus.parquet` |
| `03_embed.py` | `data/corpus.parquet` | `data/embeddings.npy` (+ pids, meta) |
| `04_reduce.py` | embeddings | `data/coords.parquet` |
| `05_cluster_label.py` | coords + embeddings | `data/labels.parquet`, `data/topic_names.json` |
| `06_render.py` | all of the above | `docs/index.html`, `docs/nlab_*.zip` |

```bash
uv sync
uv run python pipeline/01_fetch.py
uv run python pipeline/02_build_corpus.py
pipeline/run_embed_on_pod.sh <pod-ip> <ssh-port>      # stage 3 on a Runpod GPU pod (~20 min on an L4)
uv run python pipeline/04_reduce.py
uv run python pipeline/05_cluster_label.py --fit-only  # free: region counts per layer
uv run python pipeline/05_cluster_label.py --min-samples 2   # names regions; needs ANTHROPIC_API_KEY
uv run python pipeline/06_render.py
```

`pipeline/check_neighbors.py` prints nearest neighbours for a dozen well-known pages; run it after
stage 3 to sanity-check the embedding before spending on naming.
