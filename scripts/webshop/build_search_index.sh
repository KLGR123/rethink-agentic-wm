#!/usr/bin/env bash

# Rebuild WebShop's full-search index (search_engine/indexes) from the complete
# product file. Run from the rethink-agentic-wm repo root.

set -euo pipefail

WEBSHOP_ENV="/fs_mol/liujiarun/envs/webshop"
WEBSHOP_PYTHON="${WEBSHOP_ENV}/bin/python"
WEBSHOP_PATH="/fs_mol/liujiarun/WebShop"
WEBSHOP_FILE_PATH="/fs_mol/liujiarun/WebShop/data/items_shuffle.json"

export JAVA_HOME="${WEBSHOP_ENV}/lib/jvm"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export LD_LIBRARY_PATH="${WEBSHOP_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${WEBSHOP_ENV}/bin:${JAVA_HOME}/bin:${PATH}"

OUT_DOCS="${OUT_DOCS:-${WEBSHOP_PATH}/search_engine/resources/documents.jsonl}"
OUT_INDEX="${OUT_INDEX:-${WEBSHOP_PATH}/search_engine/indexes}"
INDEX_THREADS="${INDEX_THREADS:-16}"

echo "[build_search_index] Converting products to ${OUT_DOCS}"

"${WEBSHOP_PYTHON}" scripts/webshop/build_search_documents.py \
    --webshop-path "${WEBSHOP_PATH}" \
    --file-path "${WEBSHOP_FILE_PATH}" \
    --out "${OUT_DOCS}"

echo "[build_search_index] Indexing to ${OUT_INDEX} (threads=${INDEX_THREADS})"

"${WEBSHOP_PYTHON}" -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${WEBSHOP_PATH}/search_engine/resources" \
    --index "${OUT_INDEX}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${INDEX_THREADS}" \
    --storePositions --storeDocvectors --storeRaw

echo "[build_search_index] Done. Verify with:"
echo " ${WEBSHOP_PYTHON} ${WEBSHOP_PATH}/search_engine/lucene_searcher.py"
