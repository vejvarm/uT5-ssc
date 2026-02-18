#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
source ~/miniconda3/bin/activate ./.conda

export NEO4J_DB_ROOT="/neo4j_ssc_test"
export NEO4J_DOCKER_CONTAINER="neo4j_server_ssc_test"

# Predict script follows existing behavior: stop default containers and start SSC test ones.
docker stop rdf4j_server >/dev/null 2>&1 || true
docker stop neo4j_server >/dev/null 2>&1 || true
docker start rdf4j_server_ssc_test >/dev/null 2>&1 || true
docker start neo4j_server_ssc_test >/dev/null 2>&1 || true

mapfile -t CONFIGS < <(find configs/8_predict/clean-rq5 -type f -name '*.json' | sort)
echo "Found ${#CONFIGS[@]} predict configs in configs/8_predict/clean-rq5"

failures=0
for cfg in "${CONFIGS[@]}"; do
  echo "Running predict config: ${cfg}"
  if ! python -m seq2seq.run_seq2seq "${cfg}"; then
    echo "FAILED predict config: ${cfg}"
    failures=$((failures + 1))
  fi
done

echo "All predict configs finished. failures=${failures}"
if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
