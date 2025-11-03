# SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark
# Authors: Sithursan Sivasubramaniam, Cedric Osei-Akoto, Yi Zhang, Kurt Stockinger, Jonathan Fürst
# Contact: jonathan.fuerst@zhaw.ch
"""SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark"""


import json
import os
import pathlib
from typing import List, Generator, Any, Dict, Tuple
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from seq2seq.utils.rdf_schema_extractor import dump_kg_json_schema
from seq2seq.utils.neo4j_schema_extractor import Neo4jSchemaExtractor
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{sivasubramaniam2024sm3texttoquerysyntheticmultimodelmedical,
      title={SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark}, 
      author={Sithursan Sivasubramaniam and Cedric Osei-Akoto and Yi Zhang and Kurt Stockinger and Jonathan Fuerst},
      year={2024},
      eprint={2411.05521},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2411.05521}, 
}
"""

_DESCRIPTION = """\
 SM3-Text-to-Query provides data representations for relational databases (PostgreSQL), document stores (MongoDB), and graph databases (Neo4j and GraphDB (RDF)), allowing the evaluation across four popular query languages, namely SQL, MQL, Cypher, and SPARQL.
"""

_HOMEPAGE = "https://github.com/jf87/SM3-Text-to-Query/tree/main"

_LICENSE = "Apache License 2.0"

_DS_NAME = "sm3"
_LANG = "sparql"
# _URL = "https://www.dropbox.com/scl/fi/37117bjr1sx1a98ozqeb0/Spider4SSC.tgz?rlkey=k92gma53cd4fmmbf98m4vygur&st=k7ngbh13&dl=1"
_FILEPATH = "~/git/uT5-fine-tuning/data/synthea_clean"

class SpiderSSC(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.2.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DS_NAME,
            version=VERSION,
            description="SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", False)
        self.db_root = None        
        self.schema_extractor = None

        self.db_ext = {
            "sql": ".sqlite",
            "sparql": ".ttl",
            "cypher": ".ttl"
        }

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "lang": datasets.Value("string"),
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
                "Classes": datasets.features.Sequence(datasets.Value("string")),
                "Properties": datasets.features.Sequence(
                    {
                        "property": datasets.Value("string"),
                        "domain": datasets.features.Sequence(datasets.Value("string")),
                        "range": datasets.features.Sequence(datasets.Value("string"))
                    }
                ),
                "NodeLabels": datasets.features.Sequence(datasets.Value("string")),
                "NodeProperties": [
                    {
                        "nodeName": datasets.Value("string"),
                        "propertyName": datasets.Value("string"),
                        "propertyTypes": datasets.features.Sequence(datasets.Value("string"))
                    }
                ],
                "RelationshipLabels": datasets.features.Sequence(datasets.Value("string")),
                "Relationships": [
                    {
                        "startNodeLabels": datasets.features.Sequence(datasets.Value("string")),
                        "relationshipType": datasets.Value("string"),
                        "endNodeLabels": datasets.features.Sequence(datasets.Value("string"))
                    }
                ],
                "RelationshipProperties": [
                    {
                        "relName": datasets.Value("string"),
                        "propertyName": datasets.Value("string"),
                        "propertyTypes": datasets.features.Sequence(datasets.Value("string"))
                    }
                ],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = _FILEPATH

        self.db_root = pathlib.Path(downloaded_filepath).joinpath("database")
        self.neo4j_extractor = Neo4jSchemaExtractor(db_root=self.db_root)
        self.schema_extractor = {
            "sql": dump_db_json_schema,
            "sparql": dump_kg_json_schema,
            "cypher": self.neo4j_extractor.dump_neo4j_schema
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "train.json")],
                    "db_path": os.path.join(downloaded_filepath, "database"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "dev.json")],
                    "db_path": os.path.join(downloaded_filepath, "database"),
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider_joint = json.load(f)
                idx = 0
                for sample in spider_joint:
                    lang = _LANG
                    query = sample[lang]
                    db_id = sample["db_id"]
                    if lang not in self.schema_cache.keys():
                        self.schema_cache[lang] = dict()

                    if db_id not in self.schema_cache[lang].keys():
                        self.schema_cache[lang][db_id] = self.schema_extractor[lang](
                            db=os.path.join(db_path, db_id, f"{db_id}{self.db_ext[lang]}"), f=db_id
                        )
                    schema = self.schema_cache[lang][db_id]
                    data_dict = {
                        "lang": lang,
                        "query": query,
                        "question": sample["question"],
                        "db_id": db_id,
                        "db_path": db_path
                    }
                    if lang == "sql":
                        data_dict.update({
                            "db_table_names": schema["table_names_original"],
                            "db_column_names": [
                                {"table_id": table_id, "column_name": column_name}
                                for table_id, column_name in schema["column_names_original"]
                            ],
                            "db_column_types": schema["column_types"],
                            "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                            "db_foreign_keys": [
                                {"column_id": column_id, "other_column_id": other_column_id}
                                for column_id, other_column_id in schema["foreign_keys"]
                            ]
                        })
                    elif lang == "sparql":
                        data_dict.update({
                            "Classes": schema.get("Classes", []),
                            "Properties": [
                                {
                                    "property": k,
                                    "domain": v["domain"],
                                    "range": v["range"]
                                }
                                for k, v in schema.get("Properties", dict()).items()
                            ]
                        })
                    elif lang == "cypher":
                        data_dict.update({
                            "NodeLabels": schema.get("NodeLabels", []),
                            "NodeProperties": [
                                {
                                    "nodeName": key, 
                                    "propertyName": val["propertyName"], 
                                    "propertyTypes": val["propertyTypes"]
                                } for key, vals in schema["NodeProperties"].items() 
                                for val in vals],
                            "RelationshipLabels": schema.get("RelationshipLabels", []),
                            "Relationships": schema.get("Relationships", []),
                            "RelationshipProperties": [{"relName": key, "propertyName": prop["propertyName"], "propertyTypes": prop["propertyTypes"]} for key, properties in schema["RelationshipProperties"].items() for prop in properties]
                        })
                    else:
                        raise NotImplementedError(f"Lang `{lang}` not supported.")

                    struct = idx, data_dict
                    idx += 1
                    yield struct 
