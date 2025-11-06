from functools import partial
import json
import random
import pathlib
import re
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataArguments, DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.helpers import log
from seq2seq.utils.rdf_schema_extractor import serialize_sparql_schema
from seq2seq.utils.neo4j_schema_extractor import serialize_cypher_schema
from seq2seq.utils.cypher_identifier_mapping import (
    CypherIdentifierMappingBuilder,
    IdentifierMapping,
    normalize_cypher_schema,
)
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction

from seq2seq.utils.helpers import replace_custom_datatypes

from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from typing import List, Dict

LOG_FILE_NAME = "spider.txt"


def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
    lowercase_query: bool,
    capitalize_query: bool
) -> str:
    _normalize = partial(normalize, to_lower=lowercase_query, capitalize=capitalize_query) if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_add_serialized_schema(ex: dict, data_args: DataArguments, data_training_args: DataTrainingArguments) -> dict:
    result: dict = {}
    lang = ex.get("lang", data_args.dataset) 
    if "sparql" in lang:
        serialized_schema = serialize_sparql_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            classes=ex.get("Classes", []),
            properties=ex.get("Properties", dict()),
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_randomized=data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            normalize_query=data_training_args.normalize_query,
            prefix=data_args.sparql_prefix_default
        )
    elif "cypher" in lang:
        normalized_schema = normalize_cypher_schema(
            ex,
            data_training_args.cypher_remove_uri_from_schema,
            data_training_args.cypher_remove_foreign_key_attributes_from_schema,
            data_training_args.cypher_normalize_data_types,
        )
        serialized_schema = serialize_cypher_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            schema=normalized_schema,
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_randomized=data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            normalize_query=data_training_args.normalize_query,
            prefix=data_args.cypher_prefix_default
        )
        mapping_builder = CypherIdentifierMappingBuilder(
            keep_collisions=data_training_args.cypher_identifier_mapping_keep_collisions
        )
        identifier_mapping = mapping_builder.build(
            schema=normalized_schema,
            strategy=data_training_args.cypher_identifier_mapping_strategy,
        )
        serialized_schema = identifier_mapping.shorten_schema(serialized_schema)
        query_short = identifier_mapping.shorten_query(ex["query"])
        mapping_payload = identifier_mapping.to_serializable()
        collision_payload = {
            ctx: [{"original": original, "short": short} for original, short in collisions]
            for ctx, collisions in identifier_mapping.collisions_by_context.items()
            if collisions
        }
        result.update({
            "serialized_schema": serialized_schema,
            "query_short": query_short,
            "cypher_identifier_map": mapping_payload,
            "cypher_mapping_collisions": collision_payload,
        })
    else:
        serialized_schema = serialize_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex.get("db_column_names", []),
            db_table_names=ex.get("db_table_names", []),
            db_column_types=ex.get("db_column_types", []),
            db_primary_keys=ex.get("db_primary_keys", []),
            db_foreign_keys=ex.get("db_foreign_keys", []),
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_randomized=data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            normalize_query=data_training_args.normalize_query,
        )
        result["serialized_schema"] = serialized_schema

    if "serialized_schema" not in result:
        result["serialized_schema"] = serialized_schema
    if "query_short" not in result:
        result["query_short"] = ex["query"]
    return result


def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    if "lang_name" in data_training_args.source_prefix:
        if "lang" not in batch.keys():
            raise NotImplementedError("TODO: add `lang` to single language datasets as well to serve as prefix.")
        prefixes = [f"({lng}) " for lng in batch["lang"]]
    elif "postgresql" in data_training_args.source_prefix:
        batch["lang"] = [data_training_args.source_prefix for _ in range(len(batch["db_id"]))]
        prefixes = [f"({lng}) " for lng in batch["lang"]]
    elif not data_training_args.source_prefix:
        prefixes = [""]*len(batch["question"])
    else:
        raise NotImplementedError("`source_prefix` param can be either `postgresql`, `lang_name` or ''.")

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema, prefix in zip(batch["question"], batch["serialized_schema"], prefixes)
    ]

    log(f"{inputs}", "inputs.log") 

    if data_training_args.collect_token_counts:
        print("Collecting token counts...")
        # Get the inputs length and tokenized lengths
        model_inputs: dict = tokenizer(
            inputs,
            padding="max_length",
            truncation=False,
            return_overflowing_tokens=False,
        )
        lang_label = batch.get("lang", ["general"])[0]
        schema_label = data_training_args.schema_serialization_type
        
        out_path = pathlib.Path(f"results/counts/{lang_label}/{schema_label}") / "token_counts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for q, inp, ids in zip(batch["question"], inputs, model_inputs["input_ids"]):
                record = {
                    "question": q,
                    "input": inp,
                    "token_count": len(ids),
                    "truncated_tokens": len(ids) - max_source_length if len(ids) > max_source_length else 0,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Token counts for {lang_label}/{schema_label} saved to `{out_path}`")
        return model_inputs     # Skip the rest of processing

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    queries_for_loss = batch.get("query_short")
    if queries_for_loss is None:
        queries_for_loss = batch["query"]

    log(f"{queries_for_loss}", "queries_for_loss.log") 

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            lowercase_query=data_training_args.lowercase_query,
            capitalize_query=data_training_args.capitalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], queries_for_loss)
    ]

    log(f"{targets}", "targets.log")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    log(f"{labels['input_ids']}", "labels.log")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _group_predictions_by_lang(
    predictions: List[str],
    metas: List[dict]
) -> tuple[Dict[str, List[str]], Dict[str, List[dict]]]:
    """
    Regroups predictions and metadata by language using functional programming.
    
    Args:
        predictions: List of prediction strings
        metas: List of metadata dictionaries containing language info
        
    Returns:
        Tuple of (grouped_predictions, grouped_metas) where each is a dictionary
        mapping language codes to lists of predictions or metadata
    """
    if len(predictions) != len(metas):
        raise ValueError("Length mismatch between predictions and metadata")
    
    # Create sorted pairs of (prediction, meta) based on language
    sorted_pairs = sorted(
        zip(predictions, metas),
        key=lambda x: x[1]["lang"]
    )
    
    # Group by language
    grouped_data = {
        lang: list(group) for lang, group in 
        groupby(sorted_pairs, key=lambda x: x[1]["lang"])
    }
    
    # Unzip the grouped data into separate prediction and meta dictionaries
    return (
        {lang: list(map(itemgetter(0), group)) for lang, group in grouped_data.items()},
        {lang: list(map(itemgetter(1), group)) for lang, group in grouped_data.items()}
    )


def group_predictions_by_lang(
    predictions: List[str],
    metas: List[dict]
) -> tuple[Dict[str, List[str]], Dict[str, List[dict]]]:
    """
    Regroups predictions and metadata by language while preserving required data structure.
    
    Args:
        predictions: List of prediction strings
        metas: List of metadata dictionaries containing language info
        
    Returns:
        Tuple of (grouped_predictions, grouped_metas) where predictions are strings and
        metas maintain their full structure with all required fields
    """
    if len(predictions) != len(metas):
        raise ValueError("Length mismatch between predictions and metadata")
    
    grouped_predictions = defaultdict(list)
    grouped_metas = defaultdict(list)
    
    # Group while preserving all fields in metas
    for pred, meta in zip(predictions, metas):
        lang = meta["lang"]
        grouped_predictions[lang].append(pred)
        # Keep the complete meta dictionary with all required fields
        grouped_metas[lang].append({
            "lang": meta.get("lang", None),
            "query": meta["query"],
            "question": meta["question"],
            "context": meta["context"],
            "label": meta["label"],
            "db_id": meta["db_id"],
            "db_path": meta["db_path"],
            "db_table_names": meta.get("db_table_names", None) or [],
            "db_column_names": meta.get("db_column_names", None) or [],
            "db_foreign_keys": meta.get("db_foreign_keys", None) or []
        })
    
    return dict(grouped_predictions), dict(grouped_metas)


def aggregate_metrics_across_languages(
    metric_dicts: Dict[str, dict],
    sample_counts: Dict[str, int]
) -> dict:
    """
    Combines metrics from multiple languages into weighted averages.
    
    Args:
        metric_dicts: Dictionary mapping language codes to their metric dictionaries
        sample_counts: Dictionary mapping language codes to number of samples for that language
        
    Returns:
        Dictionary containing the aggregated metrics
    """
    total_samples = sum(sample_counts.values())
    aggregated_metrics = defaultdict(float)
    
    # For each language
    for lang, metrics in metric_dicts.items():
        # Weight for this language based on sample count
        weight = sample_counts[lang] / total_samples
        
        # Add weighted contribution from each metric
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregated_metrics[metric_name] += value * weight
    
    return dict(aggregated_metrics)


class SpiderTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = [
                [token if token != -100 else self.tokenizer.pad_token_id for token in label_seq]
                for label_seq in label_ids
            ]

        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        # decoded_label_ids = [d.replace("</s>", "").replace("<pad>", "").strip() for d in decoded_label_ids]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        predictions = [d.replace("</s>", "").replace("<pad>", "").replace("{{", "{").replace("}}", "}").strip() for d in predictions]
        
        logs = []
        metas = []
        predictions_new = []
        # pattern = r"\^\^[^\s:]+(?::[^\s:]+)*:"
        for x, context, label, pred in zip(examples, inputs, decoded_label_ids, predictions):
            lang = x.get("lang", None)
            processed_pred = pred
            processed_label = label
            mapping_payload = x.get("cypher_identifier_map")

            if lang == "sparql":
                processed_pred = replace_custom_datatypes(processed_pred, keep_xsd=False)
                processed_label = replace_custom_datatypes(processed_label, keep_xsd=False)
            elif lang == "cypher":
                identifier_mapping: Optional[IdentifierMapping] = None
                if mapping_payload:
                    try:
                        identifier_mapping = IdentifierMapping.from_serializable(mapping_payload)
                    except ValueError:
                        identifier_mapping = None
                if identifier_mapping is None:
                    identifier_mapping = IdentifierMapping()
                processed_pred = identifier_mapping.restore_query(processed_pred)
                processed_label = identifier_mapping.restore_query(processed_label)

            predictions_new.append(processed_pred)
            meta = {
                "lang": lang,
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": processed_label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x.get("db_table_names", []),
                "db_column_names": x.get("db_column_names", []),
                "db_foreign_keys": x.get("db_foreign_keys", []),
            }
            if lang == "cypher":
                mapping_payload = x.get("cypher_identifier_map")
                if mapping_payload:
                    meta["cypher_identifier_map"] = mapping_payload
            metas.append(meta)
            logs.append({"context": context, "pred_before": pred, "pred_after": processed_pred, "label_before": label, "label_after": processed_label})

        with open(f"{self.args.output_dir}/preds_and_labels_{stage}.json", "w") as f:
            json.dump(logs, f, indent=4)

        # metas = [
        #     {
        #         "lang": x.get("lang", None),
        #         "query": x["query"],
        #         "question": x["question"],
        #         "context": context,
        #         "label": label,
        #         "db_id": x["db_id"],
        #         "db_path": x["db_path"],
        #         "db_table_names": x.get("db_table_names", []),
        #         "db_column_names": x.get("db_column_names", []),
        #         "db_foreign_keys": x.get("db_foreign_keys", []),
        #     }
        #     for x, context, label in zip(examples, inputs, decoded_label_ids)
        # ]

        assert len(metas) == len(predictions_new)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions_new, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions_new, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]

        if metas[0].get("lang", None) is not None:
            # Print debug info
            ex_ind = random.randint(0, (len(predictions)-1)//3)
            log(f"Preds before: {predictions[ex_ind]}", LOG_FILE_NAME)
            log(f"Metas before: {metas[ex_ind]}", LOG_FILE_NAME)
            
            predictions, metas = group_predictions_by_lang(predictions, metas)
            
            # Print debug info
            for lang in predictions:
                log(f"Preds after [{lang}]: {predictions[lang][ex_ind]}", LOG_FILE_NAME)
                log(f"Metas after [{lang}]: {metas[lang][ex_ind]}", LOG_FILE_NAME)
            
            # Collect metrics for each language
            all_metrics = {}
            for lang in metas.keys():
                try:
                    metrics = self.metric.compute(
                        predictions=predictions[lang],
                        references=metas[lang],
                        lang=lang
                    )
                    if metrics is not None:
                        all_metrics[lang] = metrics
                except Exception as e:
                    print(f"Error computing metrics for language {lang}: {str(e)}")
                    continue
            
            if not all_metrics:
                return {}
                
            # Combine metrics from all languages
            metric_names = set()
            for lang_metrics in all_metrics.values():
                metric_names.update(lang_metrics.keys())
                
            final_metrics = {}
            total_samples = sum(len(metas[lang]) for lang in all_metrics.keys())
            
            for metric_name in metric_names:
                weighted_sum = 0.0
                for lang, lang_metrics in all_metrics.items():
                    print(f"{lang}: {lang_metrics[metric_name]}")
                    if metric_name in lang_metrics:
                        weight = len(metas[lang]) / total_samples
                        weighted_sum += lang_metrics[metric_name] * weight
                final_metrics[metric_name] = weighted_sum
                
            return final_metrics
        else:
            references = metas
            return self.metric.compute(predictions=predictions, references=references)
