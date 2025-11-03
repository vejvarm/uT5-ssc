import argparse
import asyncio
import json
import pathlib
from itertools import chain
from third_party.test_suite.neo4j_connector import Neo4jConnector  # Ensure your Neo4jConnector class is in this module
from seq2seq.utils.neo4j_schema_extractor import Neo4jSchemaExtractor

async def main(args):
    dataset_folder = args.dataset_folder
    split = args.split
    neo4j_root = args.neo4j_root
    keep = args.keep_existing

    db_root = dataset_folder.joinpath("database")

    assert dataset_folder.exists(), "Dataset folder does not exist!"

    dev_split = []
    train_split = []
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
    kg_names = set(entry["db_id"] for entry in chain(dev_split, train_split))

    uname = "neo4j"
    password = "secretserver"
    connector = Neo4jConnector(username=uname, password=password, neo4j_root=neo4j_root)  # Update credentials as necessary
    extractor = Neo4jSchemaExtractor(user=uname, password=password, db_root=db_root)

    # Wipe existing graphs
    await connector.wipe_databases()
    
    count = 0
    kgs_with_schema = set()
    kgs_with_error = set()
    for kg_name in kg_names:
        count += 1
        if keep and extractor.check_if_exists(kg_name):
            print(f"Skipping `{kg_name}`")
            kgs_with_schema.add(kg_name)
            continue
        try:
            print(f"Processing knowledge graph: {kg_name}")
            
            # Create a new Neo4j database
            await connector.create_database(kg_name)
            
            # Use the created database
            await connector.use_database(kg_name)
            
            # Extract prefixes and populate the database with TTL file data
            prefixes = connector.extract_prefixes_from_ttl(kg_name)
            await connector.init_database(prefixes, kg_name)

            # Extract schema from the DB
            schema = extractor.extract_schema(kg_name, restructure=True, dump=True, overwrite=True)

            print(f"Schema from `{kg_name}` extracted succesfully.")
            kgs_with_schema.add(kg_name)

            if (count+1) % 50 == 0:
                print("Max number of databases reached, wiping existing dbs.")
                await connector.wipe_databases()
            
        except Exception as e:
            print(f"Error processing {kg_name}: {e}")
            kgs_with_error.add(kg_name)
    print(f"Successfully extracted {len(kgs_with_schema)} dbs. Dbs with errors ({len(kgs_with_error)}): `{kgs_with_error}`")

    # Ensure the Neo4j connection is closed
    await connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="""
python seq2seq/extract_neo4j_schemas.py ~/git/uT5-fine-tuning/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC --split all --keep-existing
""")

    parser.add_argument("dataset_folder", type=pathlib.Path, help="Path to the root of the Spider4SSC dataset folder")
    parser.add_argument("--split", default="dev", choices=["dev", "train", "all"], help="Split for which to load the knowledge graphs")
    parser.add_argument("--neo4j-root", default=pathlib.Path("/neo4j/"), type=pathlib.Path, help="Root folder of the neo4j server on your drive")
    parser.add_argument("--keep-existing", default=False, action=argparse.BooleanOptionalAction, help="If True, only extract schemas that don't exist in the database subfolder yet.")
    asyncio.run(main(parser.parse_args()))