import json
import pathlib
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers.training_args import TrainingArguments
from seq2seq.utils.bridge_content_encoder import get_database_matches
from seq2seq.utils.helpers import log
import re
import random


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
            "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
            "value if set."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
            "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    schema_serialization_type: str = field(
        default="peteshaw",
        metadata={"help": "Choose between ``verbose`` and ``peteshaw`` schema serialization."},
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the context. Needed for Picard."},
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the database content to resolve field matches."},
    )
    normalize_query: bool = field(default=True, metadata={"help": "Whether to normalize the SQL queries."})
    target_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )
    lowercase_query: bool = field(default=False, metadata={"help": "Whether to lowercase during normalization."})
    capitalize_query: bool = field(default=False, metadata={"help": "Whether to capitalize keywords during normalization."})
    collect_token_counts: bool = field(
        default=False,
        metadata={"help": "Whether to collect token counts for the dataset."},
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to be used. Choose between ``spider``, ``cosql``, or ``cosql+spider``, or ``spider_realistic``, or ``spider_syn``, or ``spider_dk``, or ``spider_ssc_{lang}``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./seq2seq/datasets/spider",
            "cosql": "./seq2seq/datasets/cosql",
            "spider_realistic": "./seq2seq/datasets/spider_realistic",
            "spider_syn": "./seq2seq/datasets/spider_syn",
            "spider_dk": "./seq2seq/datasets/spider_dk",
            "spider_ssc_sql": "./seq2seq/datasets/ssc/spider_ssc_sql",  # spider4ssc_sql
            "spider_ssc_sparql": "./seq2seq/datasets/ssc/spider_ssc_sparql",  # spider4ssc_sparql
            "spider_ssc_cypher": "./seq2seq/datasets/ssc/spider_ssc_cypher",  # spider4ssc_cypher
            "spider_ssc_joint": "./seq2seq/datasets/ssc/spider_ssc_joint",  # spider4ssc (all at once)
            "sm3_sql": "./seq2seq/datasets/synthea/sm3_sql",
            "sm3_sparql": "./seq2seq/datasets/synthea/sm3_sparql",
            "sm3_cypher": "./seq2seq/datasets/synthea/sm3_cypher",            
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="both",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )
    #we are referencing spider_realistic to spider metrics only as both use the main spider dataset as base.
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./seq2seq/metrics/spider",
            "spider_realistic" : "./seq2seq/metrics/spider",
            "cosql": "./seq2seq/metrics/cosql",
            "spider_syn": "./seq2seq/metrics/spider",
            "spider_dk": "./seq2seq/metrics/spider",
            "spider_ssc_sql": "./seq2seq/metrics/spider",  # spider4ssc_sql 
            "spider_ssc_sparql": "./seq2seq/metrics/spidersparql",  # spider4ssc_sparql
            "spider_ssc_cypher": "./seq2seq/metrics/spidercypher",  # spider4ssc_cypher
            "spider_ssc_joint": "./seq2seq/metrics/spiderjoint",  # spider4ssc (all at once)
            "sm3_sql": "./seq2seq/metrics/spider",
            "sm3_sparql": "./seq2seq/metrics/spidersparql",
            "sm3_cypher": "./seq2seq/metrics/spidercypher",
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections : Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"}
    )

    sparql_prefix_default: Optional[str] = field(
        default="http://valuenet/ontop/",
        metadata={"help": "Default prefix which will get removed from resources in compact RDF schema representations."}
    )

    cypher_prefix_default: Optional[str] = field(
        default=None,
        metadata={"help": "Default prefix which will get removed from node names in Neo4j schema representations."}
    )
    
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to download datasets that are untrusted by default or not."}
    )


@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]


# TODO: get schemas for SPARQL and Cypher
def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex.get("db_table_names", []),
                "db_column_names": ex.get("db_column_names", []),
                "db_column_types": ex.get("db_column_types", []),
                "db_primary_keys": ex.get("db_primary_keys", []),
                "db_foreign_keys": ex.get("db_foreign_keys", []),
            }
    return schemas


def _prepare_train_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    return TrainSplit(dataset=dataset, schemas=schemas)


def _prepare_eval_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:
    if (data_training_args.max_val_samples is not None 
            and data_training_args.max_val_samples < len(dataset)):
        eval_examples = dataset.select(range(data_training_args.max_val_samples))
    else:
        eval_examples = dataset
    schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None

    print("Preparing splits")
    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_training_args=data_training_args,
                add_serialized_schema=add_serialized_schema,
                pre_process_function=pre_process_function,
            )
            for section in data_args.test_sections
        }
        test_split_schemas = {}
        for split in test_splits.values():
            test_split_schemas.update(split.schemas)

    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
        schemas=schemas
    )


def normalize(query: str, to_lower: bool = False, capitalize: bool = False) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    import re

    def extract_cypher_variables(s: str) -> set:
        """
        Extracts Cypher variables for both nodes and relationships.
        Examples of variables to catch:
          - (T1:Label)
          - (T2)
          - [r:REL]
          - [rel]
        """
        pattern = r"""
            \(\s*(?P<var_node>[A-Za-z_][A-Za-z0-9_]*)       # (T1   or (myNode
            |\[\s*(?P<var_rel>[A-Za-z_][A-Za-z0-9_]*)       # [r    or [relVar
        """
        vars_found = set()
        for m in re.finditer(pattern, s, flags=re.VERBOSE):
            if m.group("var_node"):
                vars_found.add(m.group("var_node"))
            if m.group("var_rel"):
                vars_found.add(m.group("var_rel"))
        return vars_found

    def lower_except_special(s: str) -> str:
        # First, gather all node/relationship variables
        cypher_vars = extract_cypher_variables(s)

        def replacer(match: re.Match) -> str:
            if match.group("quoted"):
                # Return quoted text as-is
                return match.group("quoted")
            elif match.group("label"):
                # Return Cypher label as-is (e.g., :ROOT_label)
                return match.group("label")
            elif match.group("namespace"):
                # Return SPARQL namespace as-is (e.g., ex:property)
                return match.group("namespace")
            else:
                # This is a "word" match.
                # If it's one of the known Cypher variables, do NOT lowercase it.
                word = match.group("word")
                if word in cypher_vars:
                    return word  # Preserve variable name's original case
                else:
                    return word.lower()

        pattern = r"""
            (?P<quoted>["'](?:\\.|[^\\"])*["']) |  # 1) Quoted text, including escaped quotes
            (?P<label>:\w+) |                     # 2) Cypher label (e.g., :ROOT_label)
            (?P<namespace>\w+:\w+) |              # 3) SPARQL namespace (e.g., ex:property)
            (?P<word>\b\w+\b)                     # 4) Other words
        """
        out = re.sub(pattern, replacer, s, flags=re.VERBOSE)
        return out

    def capitalize_keywords(s):
        # List of SQL, SPARQL, and Cypher keywords to capitalize
        keywords = [
            # SQL/SPARQL Keywords
            "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "LIMIT", "COUNT", 
            "FILTER", "OPTIONAL", "PREFIX", "DISTINCT", "UNION", "HAVING",
            
            # Cypher Keywords
            "MATCH", "RETURN", "CREATE", "MERGE", "WITH", "SET", "DELETE", 
            "DETACH", "REMOVE", "FOREACH", "CALL", "UNWIND", "LOAD CSV", 
            "USING INDEX", "DROP INDEX", "EXISTS", "EXPLAIN", "PROFILE", 
            "START", "SKIP", "UNION", "ORDER BY"
        ]

        # Ensure consistent capitalization of all keywords
        for keyword in keywords:
            s = re.sub(rf"\b{keyword.lower()}\b", keyword, s, flags=re.IGNORECASE)
        return s

    normalized_query = comma_fix(white_space_fix(query))

    if to_lower:
        normalized_query = lower_except_special(normalized_query)

    if capitalize:
        normalized_query = capitalize_keywords(normalized_query)     
    return normalized_query


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    db_column_types: List[str],
    db_primary_keys: Dict[str, int],
    db_foreign_keys: List[Dict[int, int]],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type in ["peteshaw", "norange"]:
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    elif schema_serialization_type == "compact":
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_range = "{column} ({range})"
        value_sep = " , "
    elif schema_serialization_type == "none":
        return ""
    else:
        raise NotImplementedError

    def get_col_ranges(columns: dict[str: list[int], str: list[str]], col_types, foreign_keys) -> list[str]:
        """for each column, gets either the data type or the foreign key parent name"""
        col_ranges = []
        col_names = columns["column_name"]
        for i in range(len(col_names)):
            primary_name = None
            for foreign_key_tuple in zip(foreign_keys['column_id'], foreign_keys['other_column_id']):
                foreign_id, primary_id = foreign_key_tuple
                if foreign_id == i:
                    _, primary_name = col_names[foreign_id], col_names[primary_id]

            col_range = col_types[i]
            if primary_name is not None:
                col_range = primary_name

            col_range = col_range.lower() if normalize_query else col_range

            col_ranges.append(col_range)
        # log(f"col_ranges@serialize_schema: {col_ranges}", "dataset.log")
        return col_ranges

    def get_column_str(table_name: str, column_name: str, col_range: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        log(f"\tget_column_str@serialize_schema: 'tab': {table_name}, 'col': {column_name_str}, 'range': {col_range}", "dataset.log")
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
            else:
                return column_str_without_values.format(column=column_name_str)
        elif schema_serialization_type == "compact":
            return column_str_with_range.format(column=column_name_str, range=col_range)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1], col_range=y[2]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                            get_col_ranges(db_column_names, db_column_types, db_foreign_keys)
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    # log(f"tables@serialize_schema: `{tables}`", "dataset-tables.log")
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema
