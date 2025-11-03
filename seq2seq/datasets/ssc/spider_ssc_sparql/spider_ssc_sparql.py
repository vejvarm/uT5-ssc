# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""


import json
import os
import pathlib
from typing import List, Generator, Any, Dict, Tuple
from seq2seq.utils.rdf_schema_extractor import dump_kg_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""

_DESCRIPTION = """\
Spider is a large-scale complex and cross-domain semantic parsing and text-toSQL dataset annotated by 11 college students
"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"

# _URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"

_DS_NAME = "spiderssc"
_URL = "https://www.dropbox.com/scl/fi/37117bjr1sx1a98ozqeb0/Spider4SSC.tgz?rlkey=k92gma53cd4fmmbf98m4vygur&st=k7ngbh13&dl=1"


class SpiderSSC(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DS_NAME,
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", False)        

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "lang": datasets.Value("string"),
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "Classes": datasets.features.Sequence(datasets.Value("string")),
                "Properties": datasets.features.Sequence(
                    {
                        "property": datasets.Value("string"),
                        "domain": datasets.features.Sequence(datasets.Value("string")),
                        "range": datasets.features.Sequence(datasets.Value("string"))
                    }
                )
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
        # downloaded_filepath = dl_manager.manual_dir
        # dataset_name = dl_manager._dataset_name or _DS_NAME
        # data_path = pathlib.Path(downloaded_filepath).joinpath(dataset_name)
        # if downloaded_filepath is None or not data_path.exists():
        #     print(f"Local files not found. Downloading from `{_URL}`.")
        downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, "Spider4SSC/train_sparql.json"),
                        os.path.join(downloaded_filepath, "Spider4SSC/train_others.json"),
                    ]
                    if self.include_train_others
                    else [os.path.join(downloaded_filepath, "Spider4SSC/train_sparql.json")],
                    "db_path": os.path.join(downloaded_filepath, "Spider4SSC/database"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "Spider4SSC/dev_sparql.json")],
                    "db_path": os.path.join(downloaded_filepath, "Spider4SSC/database"),
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
                spider_sparql = json.load(f)
                for idx, sample in enumerate(spider_sparql):
                    db_id = sample["db_id"]
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_kg_json_schema(
                            db=os.path.join(db_path, db_id, f"{db_id}.ttl"), f=db_id
                        )
                    schema = self.schema_cache[db_id]
                    yield idx, {
                        "lang": "sparql",
                        "query": sample["query"],
                        "question": sample["question"],
                        "db_id": db_id,
                        "db_path": db_path,
                        "Classes": schema.get("Classes", []),
                        "Properties": [
                            {
                                "property": k,
                                "domain": v["domain"],
                                "range": v["range"]
                            }
                            for k, v in schema.get("Properties", dict()).items()
                        ],
                    }
