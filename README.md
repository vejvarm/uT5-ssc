Additional information for the extension for Text-to-SPARQL and Text-to-Cypher:
---

# Run training without database serialization

## Text-to-SQL
```
python seq2seq/run_seq2seq.py configs/train_ssc_sql.json
```

## Text-to-SPARQL
1) start rdf4j server
```bash
docker run -p 8181:8080 eclipse/rdf4j-workbench:latest
```
2) serve the relevant repositories from ttl database files
```bash
python seq2seq/serve_rdf4j_graphs.py ./.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev
```
3) run training
```bash
python seq2seq/run_seq2seq.py configs/train_ssc_sparql.json
```
