"""Stage 3: embed each page's embed_text. Meant to run on a GPU pod, but runs anywhere.

Reads  data/corpus.parquet            (columns: pid, embed_text)
Writes data/embeddings.npy            float32, shape (n_pages, dim), L2-normalised, row i = corpus row i
       data/embeddings_pids.npy       int64, the pid of each row, so downstream stages can assert alignment
       data/embeddings_meta.json      model name, dim, counts

Documents are embedded with no instruction prefix: Qwen3-Embedding puts instructions on the query side
only, and this corpus is compared document-to-document. Work is done in chunks that are each written to
data/embeddings_parts/ before the next starts, so a killed run resumes from the last finished chunk.

    uv run --extra embed python pipeline/03_embed.py [--limit 200]   # smoke test on 200 pages
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
IN = DATA / "corpus.parquet"
OUT = DATA / "embeddings.npy"
OUT_PIDS = DATA / "embeddings_pids.npy"
OUT_META = DATA / "embeddings_meta.json"
PARTS = DATA / "embeddings_parts"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


def load_model(name, max_length):
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    try:
        import flash_attn  # noqa: F401

        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        print("flash_attn not installed; using the default attention implementation")
    model = SentenceTransformer(name, model_kwargs=model_kwargs, tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = max_length
    dim = model.get_sentence_embedding_dimension()
    print(f"loaded {name} on {model.device}, max_seq_length={model.max_seq_length}, dim={dim}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024, help="token cap per text; embed_text is <=400 words")
    ap.add_argument("--chunk-size", type=int, default=1024, help="pages per checkpointed chunk")
    ap.add_argument("--limit", type=int, default=None, help="only embed the first N pages (smoke test)")
    args = ap.parse_args()

    df = pd.read_parquet(IN, columns=["pid", "embed_text"])
    if args.limit:
        df = df.head(args.limit)
    texts = df.embed_text.tolist()
    pids = df.pid.to_numpy()
    n = len(texts)
    words = sum(len(t.split()) for t in texts)
    print(f"{n} texts, {words:,} words total, {words / n:.0f} words/text")

    PARTS.mkdir(parents=True, exist_ok=True)
    chunks = list(range(0, n, args.chunk_size))
    done = {i for i in chunks if (PARTS / f"part_{i:06d}.npy").exists()}
    print(f"{len(done)} of {len(chunks)} chunks already done")
    model = None
    t0 = time.time()
    for k, start in enumerate(chunks):
        if start in done:
            continue
        if model is None:
            model = load_model(args.model, args.max_length)
        batch = texts[start : start + args.chunk_size]
        emb = model.encode(batch, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        assert emb.shape[0] == len(batch) and np.isfinite(emb).all(), f"bad chunk at {start}"
        part = PARTS / f"part_{start:06d}.npy"
        tmp = part.with_suffix(".npy.tmp")
        np.save(tmp, emb)
        os.replace(tmp, part)
        elapsed = time.time() - t0
        done_n = start + len(batch)
        print(
            f"chunk {k + 1}/{len(chunks)}: {done_n}/{n} texts, {elapsed:.0f}s elapsed, {done_n / elapsed:.1f} texts/s",
            flush=True,
        )

    parts = [np.load(PARTS / f"part_{start:06d}.npy") for start in chunks]
    emb = np.concatenate(parts, axis=0)
    assert emb.shape[0] == n, f"assembled {emb.shape[0]} rows for {n} texts"
    for out, arr in ((OUT, emb), (OUT_PIDS, pids)):
        tmp = out.with_suffix(".npy.tmp")
        np.save(tmp, arr)
        os.replace(tmp, out)
    meta = {"model": args.model, "dim": int(emb.shape[1]), "n": n, "max_length": args.max_length, "normalized": True}
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT} {emb.shape} and {OUT_PIDS}; {meta}")


if __name__ == "__main__":
    main()
