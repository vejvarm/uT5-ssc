cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

# ______ON FIRST RUN______:
# # serve RDF4j server in docker
# docker run --name rdf4j_server -d -p 8181:8080 eclipse/rdf4j-workbench:latest

# # serve Neo4j server in docker
# sudo useradd -m -s /bin/bash neo4j
# sudo mkdir /neo4j/import
# sudo chown -R neo4j:neo4j /neo4j
# sudo usermod -aG docker ubuntu
# newgrp docker
# export groups=( $( id --real --groups neo4j ) )
# docker run \
#     --name neo4j_server \
#     --user="$(id -u neo4j):$(id -g neo4j)" \
#     --group-add=$groups \
#     -p7474:7474 -p7687:7687 \
#     -d \
#     -v /neo4j/data:/data \
#     -v /neo4j/logs:/logs \
#     -v /neo4j/import:/import \
#     -v /neo4j/plugins:/plugins \
#     -v /neo4j/conf:/conf \
#     --env NEO4J_PLUGINS='["apoc", "n10s"]' \
#     --env NEO4J_AUTH=neo4j/secretserver \
#     --env NEO4J_db_transaction_timeout=60s \
#     --env NEO4J_db_lock_acquisition_timeout=60s \
#     --env CLASSPATH_PREFIX=/neo4j/lib/dozerdb-plugin-5.20.0.0.jar \
#     graphstack/dozerdb:5.20.0.0-alpha.1

# # Copy database files to neo4j import folder
# sudo cp -r ~/data/git/uT5-ssc/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC/database /neo4j/import/
# sudo chown -R neo4j:neo4j /neo4j
# docker start neo4j_server

# # Populate databases
# python -m seq2seq.serve_rdf4j_graphs ~/data/git/uT5-ssc/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev
# python -m seq2seq.serve_neo4j_graphs ~/data/git/uT5-ssc/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev

# ______________________

# __ON SUBSEQUENT RUNS__
docker start rdf4j_server
docker start neo4j_server


python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_compact.json  # TODO
python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_norange.json  # TODO
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sparql_norange.json  # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sparql_compact.json  # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sql_compact.json  # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sql_no-schema.json   # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sparql_no-schema.json  # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_no-schema.json   # DONE
# python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/sql_norange.json  # DONE