"""Spider Test Suite Execution Accuracy metric."""
import json
import logging
import pathlib
from typing import Optional, Dict, Any
from third_party.test_suite import evaluation as test_suite_evaluation

logger = logging.getLogger(__name__)


def compute_test_suite_metric(predictions, references, db_dir: Optional[str] = None, lang: str = None) -> Dict[str, Any]:
    if db_dir is None:
        db_dir = references[0]["db_path"]

    foreign_key_maps = dict()
    # for reference in references:
    #     if reference["db_id"] not in foreign_key_maps:
    #         foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
    #             {
    #                 "table_names_original": reference["db_table_names"],
    #                 "column_names_original": list(
    #                     zip(
    #                         reference["db_column_names"]["table_id"],
    #                         reference["db_column_names"]["column_name"],
    #                     )
    #                 ),
    #                 "foreign_keys": list(
    #                     zip(
    #                         reference["db_foreign_keys"]["column_id"],
    #                         reference["db_foreign_keys"]["other_column_id"],
    #                     )
    #                 ),
    #             }
    #         )

    if lang is None:
        raise NotImplementedError("Metric langugage must be `sparql`")
    
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir if db_dir is not None else references[0]["db_path"],
        kmaps=foreign_key_maps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
        lang=lang
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}
    all_scores = {"exec": [], "exact": []}
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)

        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            res = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
            continue
        if "exec" in res:
            all_scores["exec"].append(res["exec"])
        if "exact" in res:
            all_scores["exact"].append(res["exact"])
    evaluator.finalize()
    return {
        "exec": evaluator.scores["all"]["exec"],
        # "all_scores": all_scores
    }
