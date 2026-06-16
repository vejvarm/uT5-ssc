from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


def load_db_id_filter(
    db_id_list: Optional[str] = None,
    db_id_file: Optional[Path] = None,
) -> Optional[Set[str]]:
    db_ids: Set[str] = set()
    if db_id_list:
        db_ids.update(part.strip() for part in db_id_list.split(",") if part.strip())
    if db_id_file:
        path = Path(db_id_file)
        db_ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return db_ids or None


def filter_records_by_db_id(
    records: Iterable[Dict],
    db_ids: Optional[Set[str]],
) -> List[Dict]:
    filtered: List[Dict] = []
    for idx, record in enumerate(records):
        if db_ids is not None and record.get("db_id") not in db_ids:
            continue
        indexed_record = dict(record)
        indexed_record["_source_record_index"] = idx
        filtered.append(indexed_record)
    return filtered
