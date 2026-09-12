"""Stage 4: reduce the embeddings to a 2-d layout with UMAP.

Reads  data/embeddings.npy, data/embeddings_pids.npy, data/corpus.parquet
Writes data/coords.parquet   (pid, x, y), row order = corpus order

This single 2-d layout is used both for plotting AND as the space Toponymy clusters in, so
min_dist is a clustering parameter here, not a cosmetic one: keep it low so regions are dense
enough for density clustering. random_state is fixed so layouts are comparable across runs
(UMAP then runs single-threaded; a few minutes for 12k x 2560).
"""

import argparse
import time

import numpy as np
import pandas as pd
import umap
from common import DATA, validate_stage_output, write_parquet_safely

OUT = DATA / "coords.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    corpus = pd.read_parquet(DATA / "corpus.parquet", columns=["pid"])
    emb = np.load(DATA / "embeddings.npy")
    pids = np.load(DATA / "embeddings_pids.npy")
    assert emb.shape[0] == len(corpus) == len(pids), (emb.shape, len(corpus), len(pids))
    assert (pids == corpus.pid.to_numpy()).all(), "embeddings are not aligned with corpus.parquet"
    print(f"embeddings {emb.shape}, aligned with corpus")

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    xy = reducer.fit_transform(emb)
    print(f"UMAP done in {time.time() - t0:.0f}s")
    df = pd.DataFrame({"pid": pids, "x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32)})
    print("x range", df.x.min(), df.x.max(), " y range", df.y.min(), df.y.max())
    validate_stage_output(df, "reduce", ["pid", "x", "y"])
    write_parquet_safely(df, OUT)


if __name__ == "__main__":
    main()
