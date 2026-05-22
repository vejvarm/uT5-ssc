import unittest
from pathlib import Path
from unittest.mock import patch

from seq2seq import analyze_target_tokenizer_roundtrip as audit
from seq2seq.utils import query_tokenizer


class FakeTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded
        self.decode_kwargs = None

    def encode(self, query, add_special_tokens):
        self.encoded_query = query
        self.add_special_tokens = add_special_tokens
        return [1, 2, 3]

    def decode(self, encoded, **kwargs):
        self.decode_kwargs = kwargs
        return self.decoded


class TargetTokenizerRoundTripTests(unittest.TestCase):
    def test_wilson_interval_for_perfect_success_is_bounded(self):
        interval = audit.wilson_interval(successes=10, total=10)

        self.assertGreater(interval["lower"], 0.7)
        self.assertEqual(1.0, interval["upper"])

    def test_summarize_token_stats_groups_by_language(self):
        rows = [
            {"language": "sql", "token_count": 2, "lexical_item_count": 1},
            {"language": "sql", "token_count": 6, "lexical_item_count": 3},
            {"language": "sparql", "token_count": 10, "lexical_item_count": 4},
        ]

        summary = audit.summarize_token_stats(rows)

        self.assertEqual(["sparql", "sql"], sorted(summary))
        self.assertEqual(2, summary["sql"]["n"])
        self.assertEqual(4.0, summary["sql"]["mean_tokens"])
        self.assertEqual(4.0, summary["sql"]["median_tokens"])
        self.assertEqual(6, summary["sql"]["p90_tokens"])
        self.assertEqual(6, summary["sql"]["max_tokens"])
        self.assertEqual(2.0, summary["sql"]["mean_tokens_per_lexical_item"])

    def test_normalize_for_match_uses_test_suite_postprocess(self):
        query = "SELECT * FROM singer WHERE age < = 20"

        self.assertEqual(
            "SELECT * FROM singer WHERE age <= 20",
            audit.normalize_for_match(query),
        )

    def test_t5_query_extra_tokens_match_training_setup(self):
        self.assertEqual(("{", "}", " <=", " <", "^^"), query_tokenizer.T5_QUERY_EXTRA_TOKENS)

    def test_clean_decoded_query_matches_evaluation_cleanup(self):
        self.assertEqual(
            "{x} <> y",
            query_tokenizer.clean_decoded_query("<pad>{{x}} < > y</s>"),
        )

    def test_audit_scores_cleaned_generated_decode(self):
        example = {
            "db_id": "concert_singer",
            "sparql": "select ?x where { ?x a :singer .}",
        }
        tokenizer = FakeTokenizer("<pad>select?x where{?x a :singer.}</s>")

        with patch.object(audit, "score_execution_equivalence", return_value=1) as metric:
            rows = audit.audit_examples(
                examples=[example],
                tokenizer=tokenizer,
                db_root=Path("data/Spider4SSC/database"),
                languages=("sparql",),
            )

        self.assertEqual("select?x where{?x a :singer.}", rows[0]["decoded_query"])
        self.assertEqual("<pad>select?x where{?x a :singer.}</s>", rows[0]["raw_decoded_query"])
        self.assertEqual({"skip_special_tokens": False}, tokenizer.decode_kwargs)
        metric.assert_called_once_with(
            db_root=Path("data/Spider4SSC/database"),
            example=example,
            language="sparql",
            decoded_query="select?x where{?x a :singer.}",
        )

    def test_audit_preserves_final_sparql_brace_after_decode_cleanup(self):
        example = {
            "db_id": "wta_1",
            "sparql": 'SELECT ?x WHERE {?x a :players .}',
        }
        tokenizer = FakeTokenizer("<pad>SELECT?x WHERE{?x a :players.}</s>")

        with patch.object(audit, "score_execution_equivalence", return_value=1):
            rows = audit.audit_examples(
                examples=[example],
                tokenizer=tokenizer,
                db_root=Path("data/Spider4SSC/database"),
                languages=("sparql",),
            )

        self.assertTrue(rows[0]["decoded_query"].endswith(".}"))

    def test_score_execution_equivalence_delegates_to_ut5_exec_eval(self):
        example = {
            "db_id": "concert_singer",
            "sql": "SELECT count(*) FROM singer",
            "sparql": "select (count( *) as ?aggregation_all) where { ?t1 a :singer . }",
            "cypher": "MATCH (t1:ROOT__singer) RETURN t1",
        }

        with patch.object(audit.exec_eval, "eval_exec_match", return_value=1) as metric:
            score = audit.score_execution_equivalence(
                db_root=Path("data/Spider4SSC/database"),
                example=example,
                language="sparql",
                decoded_query="select (count( *) as ?aggregation_all) where { ?t1 a :singer . }",
            )

        self.assertEqual(1, score)
        metric.assert_called_once_with(
            db="data/Spider4SSC/database/concert_singer/concert_singer.ttl",
            p_str="select (count( *) as ?aggregation_all) where { ?t1 a :singer . }",
            g_str=example["sparql"],
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
            lang="sparql",
        )


if __name__ == "__main__":
    unittest.main()
