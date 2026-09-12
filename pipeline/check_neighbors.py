"""Sanity-check the embedding space: print the nearest neighbours of a few well-known pages.

Neighbours should be conceptually related (adjoint functor ~ unit of an adjunction), not merely
pages that share a format. Run after stage 3, before spending anything on cluster naming.
"""

import sys

import numpy as np
import pandas as pd
from common import DATA

PROBES = [
    "adjoint functor",
    "Kan extension",
    "topological space",
    "homotopy type theory",
    "sheaf",
    "string theory",
    "Hilbert space",
    "linear logic",
    "Lie algebra",
    "measure theory",
    "quantum field theory",
    "cardinal number",
]


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet", columns=["pid", "name", "tier"])
    emb = np.load(DATA / "embeddings.npy")
    pids = np.load(DATA / "embeddings_pids.npy")
    assert (pids == corpus.pid.to_numpy()).all()
    idx = {n: i for i, n in enumerate(corpus.name)}
    probes = sys.argv[1:] or PROBES
    for name in probes:
        if name not in idx:
            print(f"\n{name!r}: not in corpus")
            continue
        sims = emb @ emb[idx[name]]
        top = np.argsort(-sims)[1:9]
        print(f"\n{name}  [{corpus.tier[idx[name]]}]")
        for j in top:
            print(f"   {sims[j]:.3f}  {corpus.name[j]}  [{corpus.tier[j]}]")


if __name__ == "__main__":
    main()
