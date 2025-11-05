import asyncio
import pathlib
import random
from neo4j import GraphDatabase
import json
from third_party.test_suite.neo4j_connector import Neo4jConnector

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)  # Call the default method for other types

class Neo4jSchemaExtractor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="secretserver", db_root: str | pathlib.Path = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        if db_root is None:
            raise NotImplementedError("path to database root folder (`db_root`) must be defined.")
        self.db_root = pathlib.Path(db_root)
        self.connector = Neo4jConnector(uri, user, password)

    def close(self):
        self.driver.close()

    def transform_db_name(self, database_name: str):
        return database_name.replace("_", "")

    def extract_node_properties(self, database: str):
        query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeLabels, propertyName, propertyTypes
        WHERE ANY(label IN nodeLabels WHERE label = "Resource")
        RETURN nodeLabels, propertyName, propertyTypes;
        """
        with self.driver.session(database=self.transform_db_name(database)) as session:
            result = session.run(query)
            return [
                {"nodeLabels": record["nodeLabels"], "propertyName": record["propertyName"], "propertyTypes": record["propertyTypes"]}
                for record in result
            ]

    def extract_relationships(self, database):
        query = """
        MATCH (start)-[r]->(end)
        RETURN DISTINCT 
            labels(start) AS startNodeLabels, 
            type(r) AS relationshipType, 
            labels(end) AS endNodeLabels
        ORDER BY relationshipType;
        """
        with self.driver.session(database=self.transform_db_name(database)) as session:
            result = session.run(query)
            return [
                {"startNodeLabels": record["startNodeLabels"], "relationshipType": record["relationshipType"], "endNodeLabels": record["endNodeLabels"]}
                for record in result
            ]

    def extract_relationship_properties(self, database):
        query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        RETURN relType, propertyName, propertyTypes;
        """
        with self.driver.session(database=self.transform_db_name(database)) as session:
            result = session.run(query)
            return [
                {"relType": record["relType"], "propertyName": record["propertyName"], "propertyTypes": record["propertyTypes"]}
                for record in result
            ]

    def extract_schema(self, db_id: str, restructure: bool = True, dump: bool = True, overwrite: bool = False, init_db: bool = False, temp_db_name: str = None):
        # Create or get the current event loop
        loop = asyncio.get_event_loop()
        # Add data to the database
        ttl_id = db_id
        if temp_db_name is not None:
            db_id = temp_db_name
        if init_db:
            loop.run_until_complete(self.connector.create_database(db_id))
            prefixes = self.connector.extract_prefixes_from_ttl(ttl_id)
            loop.run_until_complete(self.connector.init_database(prefixes, ttl_id))
            print(prefixes)
            
        node_properties = self.extract_node_properties(db_id)
        relationships = self.extract_relationships(db_id)
        relationship_properties = self.extract_relationship_properties(db_id)
        print(node_properties)

        schema = {
            "Nodes": node_properties,
            "Relationships": relationships,
            "RelationshipProperties": relationship_properties
        }
        if restructure:
            schema = self._restructure_schema(schema)

        if dump:
            self.save_schema(schema, ttl_id, overwrite)

        return schema

    def get_path(self, db_id):
        return self.db_root.joinpath(db_id).joinpath(f"{db_id}.neo4j-schema.json")

    def get_schema(self, db_id):
        """Returns existing schema for `db_id` or extracts a new from the relevant database."""
        if self.check_if_exists(db_id):
            path_to_schema = self.get_path(db_id)
            return json.load(path_to_schema.open())
        else:
            return self.extract_schema(db_id, True, True, False, init_db=False)

    def _restructure_schema(self, schema):
        # Extract node labels and properties
        node_labels = set()
        node_properties = {}
        for node in schema["Nodes"]:
            for label in node["nodeLabels"]:
                if label != "Resource":  # Skip "Resource" labels
                    node_labels.add(label)
                    if label not in node_properties:
                        node_properties[label] = []
                    node_properties[label].append({
                        "propertyName": node["propertyName"],
                        "propertyTypes": node["propertyTypes"]
                    })

        # Extract relationship labels
        relationship_labels = set()
        for rel in schema["Relationships"]:
            relationship_labels.add(rel["relationshipType"])

        # Clean relationships
        relationships = []
        for rel in schema["Relationships"]:
            rel["startNodeLabels"] = [label for label in rel["startNodeLabels"] if label != "Resource"]
            rel["endNodeLabels"] = [label for label in rel["endNodeLabels"] if label != "Resource"]
            relationships.append(rel)

        # Restructure schema
        rel_prop_dict = {}
        for entry in schema["RelationshipProperties"]:
            key = entry["relType"].strip(":`")
            if key not in rel_prop_dict.keys():
                rel_prop_dict[key] = list()
            
            if entry['propertyName'] is not None:
                rel_prop_dict[key].append({'propertyName': entry['propertyName'], 'propertyTypes': entry['propertyTypes']})

        restructured_schema = {
            "NodeLabels": list(node_labels),
            "NodeProperties": node_properties,
            "RelationshipLabels": list(relationship_labels),
            "Relationships": relationships,
            "RelationshipProperties": rel_prop_dict
        }
        return restructured_schema
    
    def check_if_exists(self, db_id: str):
        path_to_schema = self.get_path(db_id)
        return path_to_schema.exists()

    def save_schema(self, schema, db_id, overwrite: bool):
        path_to_schema = self.get_path(db_id)
        if overwrite or not self.check_if_exists(db_id):
            with open(path_to_schema, "w", encoding="utf-8") as file:
                json.dump(schema, file, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
                print(f"Saved schema to `{path_to_schema}`")


    def dump_neo4j_schema(self, db: str | pathlib.Path, f: str):
        db = pathlib.Path(db)
        db_json = db.with_suffix(".neo4j-schema.json")
        data = {
            "db_id": f 
        }
        if db_json.exists():
            data.update(json.load(db_json.open()))
        else:
            data.update(self.extract_schema(f, True, True, False, init_db=True, temp_db_name="tempdb"))
        return data


def serialize_cypher_schema(question: str, 
                            db_path: str | pathlib.Path, 
                            db_id: str, 
                            schema: dict,
                            schema_serialization_type: str = "compact",
                            schema_serialization_randomized: bool = False,
                            schema_serialization_with_db_id: bool = True,
                            schema_serialization_with_db_content: bool = False,
                            normalize_query: bool = True,
                            prefix: str = None):
    if schema_serialization_with_db_content:
        raise NotImplementedError("`schema_serialization_with_db_content==True` is not supported for RDF databases (Cypher).")
    db_id_str = f" | {db_id} | "
    if schema_serialization_type == "compact":
        node_sep = " | "
        prop_sep = " , "
        serialized_schema = serialize_schema_compactly(schema, prefix, schema_serialization_randomized, node_sep, prop_sep,
                                                       include_range=True)
    elif schema_serialization_type == "norange":
        node_sep = " | "
        prop_sep = " , "
        serialized_schema = serialize_schema_compactly(schema, prefix, schema_serialization_randomized, node_sep, prop_sep,
                                                       include_range=False)
    elif schema_serialization_type == "none":
        serialized_schema = ""
    else:
        raise NotImplementedError("`schema_serialization_type` for Cypher must be `compact`, `norange` or `none`.")

    # print(f"Serialized schema: {serialized_schema}")

    if schema_serialization_with_db_id:
        serialized_schema = db_id_str + serialized_schema

    return serialized_schema

def serialize_schema_compactly(schema: dict, prefix: str = None, shuffle: bool = True, node_sep=" | ", prop_sep=", ", include_range=True):
    """
    Generate a compact schema representation.
    Args:
        schema (dict): The restructured schema.
        prefix (str): Optional prefix to remove from class and property names.
        shuffle (bool): If true, the property/relationship name order is shuffled randomly for each node
        node_sep (str): Separator between nodes in the compact representation.
        prop_sep (str): Separator between properties/relationships within a node.
        include_range (bool): When true, data types/ranges are included after each property/relationship in brackets

    Returns:
        str: Compact representation of the schema.
    """
    compact_representation = []
    # print(schema)

    # Process NodeProperties
    for node_label in schema["NodeLabels"]:
        class_name = node_label[len(prefix):] if prefix and node_label.startswith(prefix) else node_label
        class_representation = f"{class_name}: "
        property_reprs = []
        # Collect node properties
        for entry in schema["NodeProperties"]:
            if node_label != entry["nodeName"]:
                continue 
            prop_name = entry["propertyName"]
            prop_type = entry["propertyTypes"][0] if entry["propertyTypes"] else "Unknown"
            if prefix and prop_name.startswith(prefix):
                prop_name = prop_name[len(prefix):]
            if include_range:
                property_reprs.append(f"{prop_name} ({prop_type})")
            else:
                property_reprs.append(f"{prop_name}")

        # Collect relationship properties
        for rel in schema["Relationships"]:
            if node_label in rel["startNodeLabels"]:
                rel_name = rel["relationshipType"]
                target_labels = [lbl[len(prefix):] if prefix and lbl.startswith(prefix) else lbl for lbl in rel["endNodeLabels"]]
                target_label = target_labels[0] if len(target_labels) == 1 else "Unknown"
                if include_range:
                    property_reprs.append(f"{rel_name} ({target_label})")
                else:
                    property_reprs.append(f"{rel_name}")

        if shuffle:
            random.shuffle(property_reprs)

        # Join properties and relationships
        class_representation += prop_sep.join(property_reprs)
        compact_representation.append(class_representation)

    if shuffle:
        random.shuffle(compact_representation)

    # Return the compact schema
    return node_sep.join(compact_representation)

def serialize_schema_compactly_old(schema: dict, prefix: str = None, shuffle: bool = True, node_sep=" | ", prop_sep=", ", include_range=True):
    """
    Generate a compact schema representation.
    Args:
        schema (dict): The restructured schema.
        prefix (str): Optional prefix to remove from class and property names.
        shuffle (bool): If true, the property/relationship name order is shuffled randomly for each node
        node_sep (str): Separator between nodes in the compact representation.
        prop_sep (str): Separator between properties/relationships within a node.
        include_range (bool): When true, data types/ranges are included after each property/relationship in brackets

    Returns:
        str: Compact representation of the schema.
    """
    compact_representation = []

    # Process NodeProperties
    for node_label, properties in schema["NodeProperties"].items():
        # Optionally remove prefix
        class_name = node_label[len(prefix):] if prefix and node_label.startswith(prefix) else node_label
        class_representation = f"{class_name}: "

        # Collect node properties
        property_reprs = []
        for prop in properties:
            prop_name = prop["propertyName"]
            prop_type = prop["propertyTypes"][0] if prop["propertyTypes"] else "Unknown"
            if prefix and prop_name.startswith(prefix):
                prop_name = prop_name[len(prefix):]
            if include_range:
                property_reprs.append(f"{prop_name} ({prop_type})")
            else:
                property_reprs.append(f"{prop_name}")

        # Collect relationship properties
        for rel in schema["Relationships"]:
            if node_label in rel["startNodeLabels"]:
                rel_name = rel["relationshipType"]
                target_labels = [lbl[len(prefix):] if prefix and lbl.startswith(prefix) else lbl for lbl in rel["endNodeLabels"]]
                target_label = target_labels[0] if len(target_labels) == 1 else "Unknown"
                if include_range:
                    property_reprs.append(f"{rel_name} ({target_label})")
                else:
                    property_reprs.append(f"{rel_name}")

        if shuffle:
            random.shuffle(property_reprs)

        # Join properties and relationships
        class_representation += prop_sep.join(property_reprs)
        compact_representation.append(class_representation)

    if shuffle:
        random.shuffle(compact_representation)

    # Return the compact schema
    return node_sep.join(compact_representation)


# Usage Example
if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "secretserver"
    database = "concert_singer"
    restructure = True  # Restructure and clean schema
    db_root = pathlib.Path("~/git/uT5-ssc/.cache/downloads/extracted/c702c18c8d855b7bc0a53f5b230cd5314a83d607fea4df3ad5612a557fae3dd2/Spider4SSC/database")
    extractor = Neo4jSchemaExtractor(uri, user, password, db_root=db_root)
    schema = extractor.extract_schema(database, restructure=restructure, dump=True, overwrite=True)

    # Save schema to a file
    # extractor.save_schema(schema, "neo4j_cleaned_schema.json")

    print("Cleaned Schema:")
    print(json.dumps(schema, indent=4, ensure_ascii=False))

    compact_schema = serialize_schema_compactly(schema, prefix=None, shuffle=False, node_sep=" | ", prop_sep=", ", include_range=True)

    # Print the compact schema
    print("Compact Schema:")
    print(compact_schema)

    extractor.close()
