cd ~/git/uT5-ssc
source ~/miniconda3/bin/activate ./.conda

rm -rf .cache/spider_ssc_sql
rm -rf .cache/spider_ssc_sparql
rm -rf .cache/spider_ssc_cypher

export NEO4J_DB_ROOT="/neo4j_ssc_test"
export NEO4J_DOCKER_CONTAINER="neo4j_server_ssc_test"

docker stop rdf4j_server
docker stop neo4j_server
docker start rdf4j_server_ssc_test
docker start neo4j_server_ssc_test

python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/cypher_compact.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/cypher_norange.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sparql_compact.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sql_compact.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sql_no-schema.json   
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sparql_no-schema.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/cypher_no-schema.json   
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sql_norange.json  
python seq2seq/count_tokens.py configs/9_count_tokens_of_test_set/sparql_norange.json  