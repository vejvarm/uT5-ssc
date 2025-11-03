import argparse
import asyncio
from itertools import chain
import json
import pathlib
from third_party.test_suite.rdf4j_connector import RDF4jConnector


async def main(args):
    dataset_folder = args.dataset_folder
    split = args.split

    assert dataset_folder.exists()

    dev_split = []
    train_split = []
    if split == "dev" or split == "all":
        try:
            dev_split = json.load(dataset_folder.joinpath("dev.json").open())
        except Exception as e:
            raise FileNotFoundError(f"Dev split file not found `{e}`")
        
    if split == "train" or split == "all":
        try:
            train_split = json.load(dataset_folder.joinpath("dev.json").open())
        except Exception as e:
            raise FileNotFoundError(f"Train split file not found `{e}`")
        
    kg_names = set()
    for entry in chain(dev_split, train_split):
        kg_names.add(entry["db_id"])


    rdf4j = RDF4jConnector()

    for kg_name in kg_names:
        try:
            await rdf4j.create_repository(kg_name)
            graph_uri = f"http://example.org/graph/{kg_name}"
            ttl_file_path = dataset_folder.joinpath(f"database/{kg_name}/{kg_name}.ttl")
            await rdf4j.add_data_to_named_graph(kg_name, graph_uri, ttl_file_path)
            print(f"Repository {kg_name} created and populated with `{ttl_file_path}`")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_folder", type=pathlib.Path, help="path to the root of the Spider4SSC dataset folder")
    parser.add_argument("--split", default="dev", choices=["dev", "train", "all"], help="split for which to load the knowledge graphs")
    asyncio.run(main(parser.parse_args()))