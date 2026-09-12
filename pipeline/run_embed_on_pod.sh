#!/usr/bin/env bash
# Run pipeline/03_embed.py on a Runpod GPU pod and bring the embeddings back.
#
#   pipeline/run_embed_on_pod.sh <ssh-host> <ssh-port> [ssh-key]
#
# Ships data/corpus.parquet and the embed script to the pod, installs sentence-transformers into
# the template's torch env, runs a 200-page smoke test to measure throughput, then the full run,
# then copies data/embeddings*.{npy,json} back. Re-running resumes: finished chunks on the pod
# are kept in data/embeddings_parts/ and skipped.
set -euo pipefail
HOST=${1:?ssh host}; PORT=${2:?ssh port}; KEY=${3:-$HOME/.ssh/id_ed25519}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$PORT" "root@$HOST")
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT")
REMOTE=/workspace/nlab-map

echo "== ship inputs"
"${SSH[@]}" "mkdir -p $REMOTE/data $REMOTE/pipeline"
"${SCP[@]}" data/corpus.parquet "root@$HOST:$REMOTE/data/corpus.parquet"
"${SCP[@]}" pipeline/03_embed.py "root@$HOST:$REMOTE/pipeline/03_embed.py"

echo "== install"
"${SSH[@]}" "cd $REMOTE && nvidia-smi -L && python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' \
  && pip install -q --break-system-packages 'sentence-transformers>=3.0' pandas pyarrow 2>&1 | tail -2 && python -c 'import sentence_transformers as s; print(\"sentence-transformers\", s.__version__)'"

echo "== smoke test (200 pages)"
"${SSH[@]}" "cd $REMOTE && rm -rf data/embeddings_parts && python pipeline/03_embed.py --limit 200 --chunk-size 200 2>&1 | grep -v Warning"

echo "== full run"
"${SSH[@]}" "cd $REMOTE && rm -rf data/embeddings_parts data/embeddings.npy && python pipeline/03_embed.py 2>&1 | grep -v Warning"

echo "== fetch outputs"
mkdir -p data
for f in embeddings.npy embeddings_pids.npy embeddings_meta.json; do
  "${SCP[@]}" "root@$HOST:$REMOTE/data/$f" "data/$f.part" && mv "data/$f.part" "data/$f"
done
ls -la data/embeddings*
echo "== done; remember: runpodctl pod remove <pod-id>"
