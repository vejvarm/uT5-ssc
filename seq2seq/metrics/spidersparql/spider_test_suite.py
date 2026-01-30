"""Spider Test Suite Execution Accuracy metric."""
import asyncio
import logging
import os
from typing import Optional, Dict, Any
from third_party.test_suite import evaluation as test_suite_evaluation
from third_party.test_suite import exec_eval
from third_party.test_suite import parse as test_suite_parse

logger = logging.getLogger(__name__)


def _db_file_path(db_dir: str, db_id: str, lang: str) -> str:
    ext = ".sqlite" if "sql" in lang else ".ttl"
    return os.path.join(db_dir, db_id, f"{db_id}{ext}")


def _run_exec(db_path: str, query: str, lang: str):
    return asyncio.run(exec_eval.exec_on_db(db_path, query, lang=lang))


def _exec_cross_lang(db_dir: str, db_id: str, gold_sql: str, pred_query: str, pred_lang: str) -> int:
    print(gold_sql, pred_query, pred_lang)
    if not gold_sql:
        return 0
    gold = exec_eval.postprocess(gold_sql)
    pred = exec_eval.postprocess(pred_query or "")
    try:
        gold = test_suite_parse.remove_distinct(gold, "sql")
    except Exception:
        return 0
    pred = test_suite_parse.remove_distinct(pred, pred_lang)
    order_matters = "order by" in gold.lower()

    db_sql = _db_file_path(db_dir, db_id, "sql")
    db_pred = _db_file_path(db_dir, db_id, pred_lang)

    g_flag, g_denotation = _run_exec(db_sql, gold, "sql")
    if g_flag == "exception":
        logger.warning("Gold SQL failed to execute for %s", db_id)
        return 0
    p_flag, p_denotation = _run_exec(db_pred, pred, pred_lang)
    if p_flag == "exception":
        return 0
    return 1 if exec_eval.result_eq(g_denotation, p_denotation, order_matters=order_matters) else 0


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

    needs_sql_fallback = any((not ref.get("query")) and ref.get("sql") for ref in references)
    if not needs_sql_fallback:
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

    exec_scores = []
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        if turn_idx < 0:
            continue
        gold_query = reference.get("query")
        if gold_query:
            db_path = _db_file_path(db_dir, reference["db_id"], lang)
            score = exec_eval.eval_exec_match(
                db=db_path,
                p_str=prediction,
                g_str=gold_query,
                plug_value=False,
                keep_distinct=False,
                progress_bar_for_each_datapoint=False,
                lang=lang,
            )
        else:
            score = _exec_cross_lang(db_dir, reference["db_id"], reference.get("sql"), prediction, lang)
        exec_scores.append(score)

    return {
        "exec": sum(exec_scores) / len(exec_scores) if exec_scores else 0.0,
    }
