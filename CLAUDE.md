# nlab-map

Interactive datamap of the nLab wiki. Pipeline stages live in `pipeline/`, numbered in run order;
each reads and writes files under `data/` (git-ignored). Rendered output goes to `docs/` (committed,
served by GitHub Pages).

## Stages and files

| Stage | Reads | Writes | Notes |
|---|---|---|---|
| `01_fetch.py` | GitHub mirrors | `data/raw/{nlab-content,nlab-content-html}/`, `data/raw/mirror_commits.json` | Shallow clones, pinned; `--update` to refresh |
| `02_build_corpus.py` | `data/raw/` | `data/corpus.parquet` | ~30 s; reports per-rule drop counts |
| `03_embed.py` | `corpus.parquet` | `data/embeddings.npy`, `embeddings_pids.npy`, `embeddings_meta.json` | Run on a GPU pod via `run_embed_on_pod.sh`; chunk-checkpointed |
| `04_reduce.py` | embeddings | `data/coords.parquet` | UMAP, fixed seed, `min_dist` low because clustering happens in this space |
| `05_cluster_label.py` | coords + embeddings | `data/labels.parquet`, `data/topic_names.json` | `--fit-only` is free; the full run calls Claude Haiku via `ANTHROPIC_API_KEY` |
| `06_render.py` | everything above | `docs/index.html`, `docs/data/nlab_*.zip` | DataMapPlot, externalised data |

`check_neighbors.py` prints nearest neighbours for a dozen known pages; run it after stage 3.

## Conventions

- Row order is corpus order everywhere. `embeddings_pids.npy` and the `pid` column in every parquet
  exist so each stage can assert alignment; do not sort or filter rows in a downstream stage.
- `labels.parquet` stores label layers **coarsest-first** (`label_layer_0` is the broadest). This is
  the reverse of Toponymy's in-memory order. The render stage re-derives finest-first from unique
  counts and asserts.
- Embedding text per page is: title + the page's self-description, taken from the first of Idea-like
  section, Definition, Statement, or a 25+ word preamble, plus the Definition when the lead was the
  Idea, capped at 400 words. The `tier` column records which. Pages with none are dropped, as are
  people, navigation, disambiguation, subpage, fragment ("X -- table"), SVG, empty, and meta pages.
- Expensive files (`embeddings.npy` especially) are never rewritten in place; use
  `common.write_parquet_safely` or the tmp+rename pattern.
- Math stays as itex/LaTeX in text fields; the markdown mirror is the source, never the HTML one
  (except for the "Last revised on" date).

## Tooling

`uv sync` for the core env; `uv sync --extra embed` adds torch + sentence-transformers (needed
locally for Toponymy's keyphrase embedder, and on the pod). `uv run ruff check pipeline/` before
committing.
