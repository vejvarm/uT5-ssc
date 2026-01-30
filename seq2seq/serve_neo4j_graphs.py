import argparse
import asyncio
import json
import pathlib
from itertools import chain
from third_party.test_suite.neo4j_connector import Neo4jConnector  # Ensure your Neo4jConnector class is in this module

async def main(args):
    dataset_folder = args.dataset_folder
    split = args.split
    neo4j_root = args.neo4j_root

    assert dataset_folder.exists(), "Dataset folder does not exist!"

    test_split = []
    dev_split = []
    train_split = []

    if split in {"test", "all"}:
        try:
            test_split = json.load(dataset_folder.joinpath("test.json").open())
        except Exception as e:
            raise FileNotFoundError(f"Test split file not found: {e}")
        
    if split in {"dev", "all"}:
        try:
            dev_split = json.load(dataset_folder.joinpath("dev.json").open())
        except Exception as e:
            raise FileNotFoundError(f"Dev split file not found: {e}")
        
    if split in {"train", "all"}:
        try:
            train_split = json.load(dataset_folder.joinpath("train.json").open())
        except Exception as e:
            raise FileNotFoundError(f"Train split file not found: {e}")
        
    # Extract unique knowledge graph names
    kg_names = set(entry["db_id"] for entry in chain(test_split, dev_split, train_split))

    uname = "neo4j"
    password = "secretserver"
    connector = Neo4jConnector(username=uname, password=password, neo4j_root=neo4j_root)  # Update credentials as necessary

    # Wipe existing graphs
    await connector.wipe_databases()

    for kg_name in kg_names:
        try:
            print(f"Processing knowledge graph: {kg_name}")
            
            # Create a new Neo4j database
            await connector.create_database(kg_name)
            
            # Use the created database
            await connector.use_database(kg_name)
            
            # Extract prefixes and populate the database with TTL file data
            prefixes = connector.extract_prefixes_from_ttl(kg_name)
            await connector.init_database(prefixes, kg_name)

            print(f"Database {kg_name} created and populated successfully.")
        except Exception as e:
            print(f"Error processing {kg_name}: {e}")

    # Ensure the Neo4j connection is closed
    await connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="""
python seq2seq/serve_neo4j_graphs.py ~/git/uT5-ssc/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split dev
""")

    parser.add_argument("dataset_folder", type=pathlib.Path, help="Path to the root of the Spider4SSC dataset folder")
    parser.add_argument("--split", default="dev", choices=["test", "dev", "train", "all"], help="Split for which to load the knowledge graphs")
    parser.add_argument("--neo4j-root", default=pathlib.Path("/neo4j/"), type=pathlib.Path, help="Root folder of the neo4j server on your drive")
    asyncio.run(main(parser.parse_args()))