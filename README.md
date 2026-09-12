# nlab-map

An interactive datamap of the [nLab](https://ncatlab.org) wiki: every concept page embedded, laid out
in two dimensions, and named by region.

## Pipeline

Numbered stages under `pipeline/`, each reading and writing files under `data/` (git-ignored):

| Stage | Reads | Writes |
|---|---|---|
| `01_fetch.py` | GitHub mirrors `ncatlab/nlab-content` (markdown) and `ncatlab/nlab-content-html` | `data/raw/`, `data/raw/mirror_commits.json` |
| `02_build_corpus.py` | `data/raw/` | `data/corpus.parquet` |
| `03_embed.py` | `data/corpus.parquet` | `data/embeddings.npy` (run on a GPU pod) |

```bash
uv sync
uv run python pipeline/01_fetch.py
uv run python pipeline/02_build_corpus.py
```
