"""Stage 5: cluster the 2-d layout into a hierarchy of regions and name them with Toponymy.

Reads  data/corpus.parquet, data/embeddings.npy, data/embeddings_pids.npy, data/coords.parquet
Writes data/labels.parquet     pid + label_layer_k (region name or "Unlabelled") + cluster_layer_k (int id, -1 = none)
       data/topic_names.json   the name hierarchy, layer sizes, and the settings used

Layer order ON DISK is coarsest-first (label_layer_0 = broadest regions), the reverse of Toponymy's
in-memory order, so the broadest layer keeps a stable name as the layer count varies between runs.
The render stage reverses and asserts.

Clustering runs on the 2-d coords (the same layout that is plotted, so named regions correspond to
what a viewer sees); the high-d embeddings feed exemplar selection. Keyphrases are chosen in their
own embedding space, so a small local sentence-transformer serves as Toponymy's text embedder.

    uv run python pipeline/05_cluster_label.py --fit-only          # free: print region counts per layer
    uv run python pipeline/05_cluster_label.py                     # names regions with Claude Haiku via the API
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from common import DATA, validate_stage_output, write_parquet_safely

OUT = DATA / "labels.parquet"
OUT_NAMES = DATA / "topic_names.json"
OBJECT_DESCRIPTION = "nLab wiki pages, each a short self-description of a mathematical or physical concept"
CORPUS_DESCRIPTION = (
    "the nLab, a wiki on category theory, higher category theory, homotopy theory, type theory, "
    "and their applications in mathematics and physics"
)


def load_inputs():
    corpus = pd.read_parquet(DATA / "corpus.parquet", columns=["pid", "name", "summary"])
    emb = np.load(DATA / "embeddings.npy")
    pids = np.load(DATA / "embeddings_pids.npy")
    coords = pd.read_parquet(DATA / "coords.parquet")
    assert (pids == corpus.pid.to_numpy()).all(), "embeddings not aligned with corpus"
    assert (coords.pid.to_numpy() == corpus.pid.to_numpy()).all(), "coords not aligned with corpus"
    xy = np.ascontiguousarray(coords[["x", "y"]].to_numpy(dtype=np.float32))  # numba kd-tree needs C order
    objects = [f"{n}: {s}" if s else n for n, s in zip(corpus.name, corpus.summary)]
    return corpus, emb, xy, objects


def fit_clusterer(xy, emb, args):
    from toponymy import ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText

    clusterer = ToponymyClusterer(
        min_clusters=args.min_clusters,
        min_samples=args.min_samples,
        base_min_cluster_size=args.base_min_cluster_size,
        next_cluster_size_quantile=args.next_cluster_size_quantile,
        verbose=False,
    )
    # low-d first, high-d second: the clusterer's argument order (the orchestrator's is reversed)
    clusterer.fit(clusterable_vectors=xy, embedding_vectors=emb, layer_class=ClusterLayerText)
    print("layer  regions  unlabelled")
    for i, layer in enumerate(clusterer.cluster_layers_):
        labels = layer.cluster_labels
        print(f"{i:5d}  {labels.max() + 1:7d}  {(labels < 0).mean():9.1%}   ({'finest' if i == 0 else ''})")
    return clusterer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-only", action="store_true", help="fit the clusterer and report layer sizes; no LLM calls")
    ap.add_argument("--min-clusters", type=int, default=6)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--base-min-cluster-size", type=int, default=10)
    ap.add_argument("--next-cluster-size-quantile", type=float, default=0.85)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--text-embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--async", dest="use_async", action="store_true", help="concurrent naming calls (broken in 0.5.4)")
    args = ap.parse_args()

    corpus, emb, xy, objects = load_inputs()
    print(f"{len(objects)} objects, embeddings {emb.shape}, coords {xy.shape}")
    clusterer = fit_clusterer(xy, emb, args)
    if args.fit_only:
        return
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

    import toponymy.llm_wrappers as llm_wrappers
    from sentence_transformers import SentenceTransformer
    from toponymy import Toponymy

    # AsyncAnthropicNamer (toponymy 0.5.4) fails with "Semaphore is bound to a different event loop" once
    # the second layer starts, so calls run sequentially unless --async is passed.
    factory = llm_wrappers.AsyncAnthropicNamer if args.use_async else llm_wrappers.AnthropicNamer
    namer = factory(model=args.model)
    print(f"namer: {factory.__name__}({args.model})", flush=True)
    text_embedder = SentenceTransformer(args.text_embedder)
    topics = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=text_embedder,
        clusterer=clusterer,  # pre-fitted: Toponymy reuses its layers rather than re-clustering
        object_description=OBJECT_DESCRIPTION,
        corpus_description=CORPUS_DESCRIPTION,
        verbose=True,
    )
    topics.fit(objects, embedding_vectors=emb, clusterable_vectors=xy)  # high-d first here

    names = topics.topic_names_  # finest-first in memory
    vectors = topics.topic_name_vectors_
    n_layers = len(names)
    assert len(names[0]) >= len(names[-1]), "expected topic_names_[0] to be the finest layer"
    out = pd.DataFrame({"pid": corpus.pid})
    for k in range(n_layers):  # write coarsest-first
        mem = n_layers - 1 - k
        out[f"label_layer_{k}"] = np.asarray(vectors[mem], dtype=object)
        out[f"cluster_layer_{k}"] = clusterer.cluster_layers_[mem].cluster_labels.astype(np.int32)
    for k in range(n_layers):
        col = out[f"label_layer_{k}"]
        print(
            f"layer {k} (coarsest-first): {col.nunique() - int((col == 'Unlabelled').any())} names, "
            f"{(col == 'Unlabelled').mean():.1%} unlabelled"
        )
    validate_stage_output(out, "cluster_label", ["pid", "label_layer_0"])
    write_parquet_safely(out, OUT)
    meta = {
        "layer_order": "coarsest-first (label_layer_0 is the broadest layer)",
        "n_layers": n_layers,
        "topic_names": [names[n_layers - 1 - k] for k in range(n_layers)],
        "settings": vars(args),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT_NAMES.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_NAMES}")


if __name__ == "__main__":
    main()
