cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

# SELECT THE DATABASE ROOT AND OVERRIDE THE KG NAME for all eval samples
# export NEO4J_DB_ROOT="/neo4j_ssc_test"
# export NEO4J_DOCKER_CONTAINER="neo4j_server_ssc_test"
# Optional: use a separate import folder for test DBs
# export NEO4J_DB_SUBFOLDER="import/database_test"
# export NEO4J_OVERRIDE_KG="ssc_test"

# ______ON FIRST RUN______:
# # serve RDF4j server in docker
# docker run --name rdf4j_server_ssc_test -d -p 8181:8080 eclipse/rdf4j-workbench:latest

# # serve Neo4j server in docker
# export groups=( $( id --real --groups neo4j ) )
# docker run \
#     --name neo4j_server_ssc_test \
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

# Copy database files to neo4j_ssc_test import folder (test DBs live in database_test)
# sudo cp -r /home/vejvar-martin-nj/git/uT5-ssc/data/ssc_test/database /neo4j_ssc_test/import/database
# sudo chown -R neo4j:neo4j /neo4j_ssc_test
# docker start neo4j_server_ssc_test

# sudo chmod 777 /neo4j_ssc_test/import/database

# # Populate databases
# python -m seq2seq.serve_rdf4j_graphs /home/vejvar-martin-nj/git/uT5-ssc/data/ssc_test --split test
# python -m seq2seq.serve_neo4j_graphs /home/vejvar-martin-nj/git/uT5-ssc/data/ssc_test --split test --neo4j-root /neo4j_ssc_test/

# Extract database schemas
# python -m seq2seq.utils.rdf_schema_extractor test --ds-path /home/vejvar-martin-nj/git/uT5-ssc/data/ssc_test --clean --dump-schema --overwrite
# python seq2seq/extract_neo4j_schemas.py /home/vejvar-martin-nj/git/uT5-ssc/data/ssc_test --neo4j-root /neo4j_ssc_test/ --split test 


# ______________________

# __ON SUBSEQUENT RUNS__
docker stop rdf4j_server_ssc_test
docker stop neo4j_server_ssc_test
docker start rdf4j_server
docker start neo4j_server

# Run predictions on test set with uT5-clean models
# clean
# python -m seq2seq.run_seq2seq configs/8_eval/clean/cypher_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/cypher_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sparql_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sparql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sql_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sql_no-schema.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean/sparql_no-schema.json 
# python -m seq2seq.run_seq2seq configs/8_eval/clean/cypher_no-schema.json 

# # dirty
python -m seq2seq.run_seq2seq configs/8_eval/dirty/sql_compact.json
python -m seq2seq.run_seq2seq configs/8_eval/dirty/sparql_compact.json
python -m seq2seq.run_seq2seq configs/8_eval/dirty/cypher_compact.json

# # injected10p
# python -m seq2seq.run_seq2seq configs/8_eval/injected10p/sql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/injected10p/sparql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/injected10p/cypher_compact.json

# # injected30p
# python -m seq2seq.run_seq2seq configs/8_eval/injected30p/sql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/injected30p/sparql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/injected30p/cypher_compact.json

# # cypher extras
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_long/cypher_compact_1024.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_long/cypher_compact_2048.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_long/cypher_norange_1024.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_long/cypher_norange_2048.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_long/cypher_norange_2048.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_short/cypher_compact_short.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_short/cypher_compact_strip_root_only.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_short/cypher_norange_short.json
# python -m seq2seq.run_seq2seq configs/8_eval/cypher_short/cypher_norange_strip_root_only.json

# joint clean RQ5
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/cypher_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/cypher_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sparql_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sparql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sql_norange.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sql_compact.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sql_no-schema.json
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/sparql_no-schema.json 
# python -m seq2seq.run_seq2seq configs/8_eval/clean-rq5/cypher_no-schema.json 
