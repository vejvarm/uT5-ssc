#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
source ~/miniconda3/bin/activate ./.conda

unset NEO4J_DB_ROOT
unset NEO4J_DOCKER_CONTAINER

# Eval script follows existing behavior: stop SSC test containers and start default ones.
docker stop rdf4j_server_ssc_test >/dev/null 2>&1 || true
docker stop neo4j_server_ssc_test >/dev/null 2>&1 || true
docker start rdf4j_server >/dev/null 2>&1 || true
docker start neo4j_server >/dev/null 2>&1 || true

mapfile -t CONFIGS < <(find configs/11_800_eval -type f -name '*.json' | sort)
echo "Found ${#CONFIGS[@]} eval configs in configs/11_800_eval"

failures=0
for cfg in "${CONFIGS[@]}"; do
  echo "Running eval config: ${cfg}"
  if ! python -m seq2seq.run_seq2seq "${cfg}"; then
    echo "FAILED eval config: ${cfg}"
    failures=$((failures + 1))
  fi
done

echo "All eval configs finished. failures=${failures}"
if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
