cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

# SELECT THE DATABASE ROOT AND OVERRIDE THE KG NAME for all eval samples
export NEO4J_DB_ROOT="/neo4j_sm3"
export NEO4J_OVERRIDE_KG="sm3"

# ______ON FIRST RUN______:
# # serve RDF4j server in docker
# docker run --name rdf4j_server_sm3 -d -p 8181:8080 eclipse/rdf4j-workbench:latest

# # serve Neo4j server in docker
# export groups=( $( id --real --groups neo4j ) )
# docker run \
#     --name neo4j_server_sm3 \
#     --user="$(id -u neo4j):$(id -g neo4j)" \
#     --group-add=$groups \
#     -p7474:7474 -p7687:7687 \
#     -d \
#     -v $NEO4J_DB_ROOT/data:/data \
#     -v $NEO4J_DB_ROOT/logs:/logs \
#     -v $NEO4J_DB_ROOT/import:/import \
#     -v $NEO4J_DB_ROOT/plugins:/plugins \
#     -v $NEO4J_DB_ROOT/conf:/conf \
#     --env NEO4J_PLUGINS='["apoc", "n10s"]' \
#     --env NEO4J_AUTH=neo4j/secretserver \
#     --env NEO4J_db_transaction_timeout=600s \
#     --env NEO4J_db_lock_acquisition_timeout=600s \
#     --env CLASSPATH_PREFIX=/neo4j/lib/dozerdb-plugin-5.20.0.0.jar \
#     --env NEO4J_server_memory_heap_initial__size=2G \
#     --env NEO4J_server_memory_heap_max__size=8G \
#     --env NEO4J_server_memory_pagecache_size=4G \
#     --env NEO4J_db_memory_transaction_total_max=10G \
#     graphstack/dozerdb:5.20.0.0-alpha.1

# # Populate databases
# python seq2seq/serve_rdf4j_graphs.py /home/vejvar-martin-nj/git/uT5-ssc/data/synthea_clean --split dev
# python seq2seq/serve_neo4j_graphs.py /home/vejvar-martin-nj/git/uT5-ssc/data/synthea_clean --split dev --neo4j-root /neo4j_sm3/


# Extract database schemas
# python seq2seq/utils/rdf_schema_extractor.py dev --ds-path /home/vejvar-martin-nj/git/uT5-ssc/data/synthea_clean --clean --dump-schema
# python seq2seq/extract_neo4j_schemas.py /home/vejvar-martin-nj/git/uT5-ssc/data/synthea_clean --neo4j-root /neo4j_sm3/ --split dev --keep-existing 


# ______________________

# __ON SUBSEQUENT RUNS__
docker start rdf4j_server_sm3
docker start neo4j_server_sm3


python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/cypher_compact.json  # TODO 300
python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sql_compact.json  # TODO 300
python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sparql_compact.json  # TODO 300
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sql_norange.json  # TODO
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sparql_norange.json  # TODO
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/cypher_norange.json  # TODO
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sql_no-schema.json   # TODO
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/sparql_no-schema.json   # TODO
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/cypher_no-schema.json   # TODO

# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/eval/sparql_compact.json
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/eval/cypher_compact.json
# python -m seq2seq.run_seq2seq configs/sm3/5_ut5-ep19-clean/eval/sql_compact.json