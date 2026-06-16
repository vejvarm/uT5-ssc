import tempfile
import unittest
from pathlib import Path

from seq2seq.utils.db_id_filter import filter_records_by_db_id, load_db_id_filter
from seq2seq.merge_exec_analysis_chunks import merge_chunk_files


class DbIdFilterTests(unittest.TestCase):
    def test_load_db_id_filter_combines_file_and_csv_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "db_ids.txt"
            path.write_text("activity_1\n\nbaseball_1\n", encoding="utf-8")

            db_ids = load_db_id_filter(db_id_list="cinema, city_record", db_id_file=path)

        self.assertEqual(
            db_ids,
            {"activity_1", "baseball_1", "cinema", "city_record"},
        )

    def test_filter_records_by_db_id_preserves_source_index(self):
        records = [
            {"db_id": "activity_1", "value": 0},
            {"db_id": "baseball_1", "value": 1},
            {"db_id": "cinema", "value": 2},
        ]

        filtered = filter_records_by_db_id(records, {"baseball_1", "cinema"})

        self.assertEqual([r["value"] for r in filtered], [1, 2])
        self.assertEqual([r["_source_record_index"] for r in filtered], [1, 2])

    def test_merge_chunk_files_restores_original_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_a = Path(tmpdir) / "chunk_a.json"
            chunk_b = Path(tmpdir) / "chunk_b.json"
            chunk_a.write_text(
                '[{"_source_record_index": 2, "value": "c"}]',
                encoding="utf-8",
            )
            chunk_b.write_text(
                '[{"_source_record_index": 0, "value": "a"},'
                '{"_source_record_index": 1, "value": "b"}]',
                encoding="utf-8",
            )

            merged = merge_chunk_files([chunk_a, chunk_b])

        self.assertEqual([record["value"] for record in merged], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
