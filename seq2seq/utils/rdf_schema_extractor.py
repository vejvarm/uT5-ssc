import json
import pathlib
import random
from copy import deepcopy
from rdflib import Graph, RDF, RDFS, Namespace, URIRef
from tqdm import tqdm

# These prefixes will be ignored
IMPLICIT_PREFIXES = ['brick', 'csvw', 'dc', 'dcat', 'dcmitype', 'dcterms', 'dcam', 'doap', 'foaf', 
                     'geo', 'odrl', 'org', 'prof', 'prov', 'qb', 'schema', 'sh', 'skos', 'sosa', 
                     'ssn', 'time', 'vann', 'void', 'wgs', 'owl', 'rdf', 'rdfs', 'xsd', 'xml']

LITERAL_RANGE_TOKENS = {
    "literal",
    "string",
    "str",
    "int",
    "integer",
    "float",
    "double",
    "decimal",
    "num",
    "number",
    "bool",
    "boolean",
    "date",
    "datetime",
    "time",
    "timestamp",
}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)  # Call the default method for other types

def dump_kg_json_schema(db: str | pathlib.Path, f: str):
    db = pathlib.Path(db)
    db_json = db.with_suffix(".rdf-schema.json")
    data = {
        "db_id": f 
    }
    if db_json.exists():
        data.update(json.load(db_json.open()))
    else:
        data.update(extract_rdf_schema(db, True, True, False))

    return data


def extract_rdf_schema(ttl_file, clean: bool, dump: bool, overwrite: bool) -> dict:
    # Load the TTL file into an RDF graph
    g = Graph(store="Oxigraph")
    g.parse(ttl_file, format="turtle")

    # Initialize data structures
    schema = {
        "Prefixes": {},
        "Classes": set(),
        "Properties": {},
        "Domains": {},
        "Ranges": {}
    }

    # Extract prefixes
    for prefix, namespace in g.namespaces():
        schema["Prefixes"][prefix] = str(namespace)

    # Identify classes by finding all unique rdf:type values used in the data
    for s, p, o in g.triples((None, RDF.type, None)):
        schema["Classes"].add(str(o))

    # Identify properties and infer domains and ranges from data patterns
    for s, p, o in g.triples((None, None, None)):
        prop_str = str(p)
        
        # Skip class type definitions
        if p == RDF.type:
            continue

        # Initialize property in schema if not already
        if prop_str not in schema["Properties"]:
            schema["Properties"][prop_str] = {"domain": set(), "range": set()}

        # Infer domain: the type of the subject
        for subject_type in g.objects(s, RDF.type):
            schema["Properties"][prop_str]["domain"].add(str(subject_type).split("#")[-1])

        # Infer range: the type of the object, or its datatype if it's a literal
        if isinstance(o, Namespace):
            for object_type in g.objects(o, RDF.type):
                schema["Properties"][prop_str]["range"].add(str(object_type).split("#")[-1])
        elif isinstance(o, URIRef):
            classes = set()
            for object_type in g.objects(o, RDF.type):  # Check its rdf:type
                classes.add(str(object_type).split("#")[-1])
            if not classes:
                classes.add("Resource")  # Mark as Resource if no specific type is found
            schema["Properties"][prop_str]["range"].update(classes)
        elif hasattr(o, "datatype"):
            schema["Properties"][prop_str]["range"].add(str(o.datatype).split("#")[-1] if o.datatype else "Literal")
        else:
            schema["Properties"][prop_str]["range"].add("Literal")

    # Flatten domains and ranges to readable strings
    for prop in schema["Properties"]:
        schema["Domains"][prop] = ", ".join(schema["Properties"][prop]["domain"])
        schema["Ranges"][prop] = ", ".join(schema["Properties"][prop]["range"])

    if clean:
        schema = _clean_schema(schema)

    if dump:
        schema_file = ttl_file.with_suffix(".rdf-schema.json")
        if overwrite or not schema_file.exists():
            print(f"Saving `{schema_file}`.")
            json.dump(schema, schema_file.open("w"), indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    return schema


def serialize_sparql_schema(question: str, 
                            db_path: str | pathlib.Path, 
                            db_id: str, 
                            classes: list, 
                            properties: dict,
                            schema_serialization_type: str = "compact",
                            schema_serialization_randomized: bool = False,
                            schema_serialization_with_db_id: bool = True,
                            schema_serialization_with_db_content: bool = False,
                            normalize_query: bool = True,
                            prefix: str = "http://valuenet/ontop/"
                            ) -> str:
    if schema_serialization_with_db_content:
            raise NotImplementedError("`schema_serialization_with_db_content==True` is not supported for RDF databases (SPARQL).")
    db_id_str = f" | {db_id} | "
    if schema_serialization_type == "compact":
        class_sep = " | "
        prop_sep = " , "
        schema = {"Classes": classes, "Properties": properties}
        serialized_schema = serialize_schema_compactly(
            schema,
            prefix,
            schema_serialization_randomized,
            class_sep,
            prop_sep,
            include_property_range=True,
            include_relationship_range=True,
        )
    elif schema_serialization_type == "norange":
        class_sep = " | "
        prop_sep = " , "
        schema = {"Classes": classes, "Properties": properties}
        serialized_schema = serialize_schema_compactly(
            schema,
            prefix,
            schema_serialization_randomized,
            class_sep,
            prop_sep,
            include_property_range=False,
            include_relationship_range=False,
        )
    elif schema_serialization_type == "hybrid":
        class_sep = " | "
        prop_sep = " , "
        schema = {"Classes": classes, "Properties": properties}
        serialized_schema = serialize_schema_compactly(
            schema,
            prefix,
            schema_serialization_randomized,
            class_sep,
            prop_sep,
            include_property_range=False,
            include_relationship_range=True,
        )
    elif schema_serialization_type == "none":
        serialized_schema = ""  
    else:
        raise NotImplementedError("`schema_serialization_type` for SPARQL must be `compact`, `norange`, `hybrid`, or `none`.")

    if schema_serialization_with_db_id:
        serialized_schema = db_id_str + serialized_schema

    return serialized_schema

def serialize_schema_compactly(
    schema,
    prefix: str = None,
    shuffle: bool = True,
    class_sep = " | ",
    prop_sep = ", ",
    include_property_range: bool = True,
    include_relationship_range: bool = True,
) -> str:
    # Initialize the compact representation
    compact_representation = []

    # Iterate through classes and their properties
    for cls in schema["Classes"]:
        class_line = cls.split("#")[-1] + ": "  # Use the local name of the class
        if prefix is not None:
            class_line = class_line.removeprefix(prefix)        
        properties = []

        prop_entries = zip(schema["Properties"]["property"], schema["Properties"]["domain"], schema["Properties"]["range"])

        for entry in prop_entries:
            prop, domains, ranges = entry
            # print(prop, domains, ranges, cls.split("#")[-1], cls)
            for domain in domains:
                for range_ in ranges:
                    if domain == cls.split("#")[-1]:  # Property belongs to the class
                        # print(domain, cls.split("#")[-1])
                        prop_name = prop.split("#")[-1]  # Use the local name of the property
                        range_name = range_.split("#")[-1] if range_ else ""
                        normalized_range = range_name.lower()
                        if prefix is not None and range_name:
                            range_name = range_name.removeprefix(prefix)
                        is_literal = not normalized_range or normalized_range in LITERAL_RANGE_TOKENS
                        should_include = include_property_range if is_literal else include_relationship_range
                        if should_include and range_name:
                            properties.append(f"{prop_name} ({range_name})")
                        else:
                            properties.append(f"{prop_name}")

        if properties:
            if shuffle:
                random.shuffle(properties)
            class_line += prop_sep.join(properties)
            compact_representation.append(class_line)

    if shuffle:
        random.shuffle(compact_representation)

    return class_sep.join(compact_representation)


def _serialize_schema_compactly_old(schema, prefix: str = None, shuffle: bool = True) -> str:
    # Initialize the compact representation
    compact_representation = []

    # Iterate through classes and their properties
    for cls in schema["Classes"]:
        class_line = cls.split("#")[-1] + ": "  # Use the local name of the class
        if prefix is not None:
            class_line = class_line.removeprefix(prefix)        
        properties = []

        for prop, domain in schema["Domains"].items():
            if domain == cls.split("#")[-1]:  # Property belongs to the class
                range_ = schema["Ranges"].get(prop, "")
                prop_name = prop.split("#")[-1]  # Use the local name of the property
                range_name = range_.split("#")[-1] if range_ else ""
                properties.append(f"{prop_name} ({range_name})")

        if properties:
            if shuffle:
                random.shuffle(properties)
            class_line += ", ".join(properties)
            compact_representation.append(class_line)

        if shuffle:
            random.shuffle(compact_representation)

    return "\n".join(compact_representation)


def _clean_schema(schema: dict, prefixes = IMPLICIT_PREFIXES) -> dict:
    popped = [schema["Prefixes"].pop(k, None) for k in prefixes]
    # print(schema["Prefixes"])
    return schema

def merge(dataset: list[dict], schemas: dict):
    merged = []
    for entry in dataset:
        kg = entry["db_id"]
        schema = _clean_schema(schemas[kg])
        entry_cp = deepcopy(entry)
        entry_cp.update(schema)
        merged.append(entry_cp)
    return merged


def main(args):
    # Example usage:
    split = args.split
    ds_path = args.ds_path
    clean_schema = args.clean
    dump_schema = args.dump_schema
    overwrite = args.overwrite
    json_path = ds_path.joinpath(f"{split}.json")
    db_folder = ds_path.joinpath("database")
    ds_data = json.load(json_path.open())
    kg_list = list({e["db_id"] for e in ds_data})
    print(kg_list)
    
    full_schemas_path = json_path.with_suffix(".schemas.json")
    full_merged_path = json_path.with_suffix(".merged.json")

    if full_schemas_path.exists() and not overwrite:
        print(f"WARNING SchemaFileFound: `{full_schemas_path}` already exists. Make sure to move/delete to recreate it.")
        schemas = json.load(full_schemas_path.open())
    else:
        full_schemas_path.unlink(missing_ok=True)
        schemas = dict()
        pbar = tqdm(kg_list, desc="Extracting schemas")
        for kg in pbar: 
            pbar.set_postfix({'kg': kg})
            full_kg_path = db_folder.joinpath(f"{kg}/{kg}.ttl")
            if kg not in schemas:
                schemas[kg] = extract_rdf_schema(full_kg_path, clean_schema, dump_schema, overwrite=overwrite)
                # compact_schema = serialize_schema_compactly(schemas[kg])
                # print(compact_schema)

        for entry in ds_data:
            entry.update(schemas[kg])

        json.dump(schemas, full_schemas_path.open("w"), indent=4, ensure_ascii=False, cls=CustomJSONEncoder)

    # Merge with the original file
    merged = merge(ds_data, schemas)
    if full_merged_path.exists() and not overwrite:
        print(f"WARNING MergedFileFound: `{full_merged_path}` already exists. Make sure to move/delete to recreate it.")
    else:
        json.dump(merged, full_merged_path.open("w"), indent=4, ensure_ascii=False, cls=CustomJSONEncoder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("split", type=str, choices=["test", "dev", "train"], help="Which split should have the schemas extracted and merged with the json")
    parser.add_argument("--ds-path", type=pathlib.Path, default=pathlib.Path("/media/freya/kubuntu-data/git/Spider4SSC/data/Spider4SSC/"), help="Path to the Spider4SSC dataset root.")
    parser.add_argument("--clean", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Clean up the schema prefixes to remove implicit ones.")
    parser.add_argument("--dump-schema", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Dump individual schema file to the location of the ttl file.")
    parser.add_argument("--overwrite", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing schema files.")
    main(parser.parse_args())
