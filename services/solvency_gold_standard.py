from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .solvency_pdf_locator import PageMatch
from .solvency_table_extractor import ExtractedTable


def load_gold_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("cases"), list):
        raise ValueError("金标准manifest缺少cases数组。")
    return manifest


def pdf_sha256(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def find_gold_case(pdf_bytes: bytes, manifest: dict) -> dict | None:
    digest = pdf_sha256(pdf_bytes)
    return next(
        (
            case
            for case in manifest.get("cases", [])
            if str(case.get("sha256", "")).lower() == digest
        ),
        None,
    )


def _compact(value: object) -> str:
    return re.sub(r"[\s：:（）()、，,。·—\\/_\-％%]", "", str(value or ""))


def evaluate_locator(
    case: dict,
    matches: Iterable[PageMatch],
) -> pd.DataFrame:
    predicted = {item.table_id: sorted(set(item.pages)) for item in matches}
    records: list[dict] = []
    for table_id, expected in case.get("tables", {}).items():
        gold_pages = sorted(set(int(item) for item in expected.get("pages", [])))
        predicted_pages = predicted.get(table_id, [])
        gold_set, predicted_set = set(gold_pages), set(predicted_pages)
        overlap = gold_set & predicted_set
        recall = len(overlap) / len(gold_set) if gold_set else 1.0
        precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
        records.append({
            "阶段": "页码定位",
            "目标表ID": table_id,
            "金标准页码": ",".join(map(str, gold_pages)),
            "实际页码": ",".join(map(str, predicted_pages)),
            "页码召回率": recall,
            "页码准确率": precision,
            "是否完全一致": predicted_pages == gold_pages,
            "项目召回率": None,
            "关键数值准确率": None,
        })
    return pd.DataFrame(records)


def _row_label(row: list[str]) -> str:
    if not row:
        return ""
    if re.fullmatch(r"\d+(?:\.\d+)*\*?", str(row[0]).strip()) and len(row) > 1:
        return str(row[1]).strip()
    return str(row[0]).strip()


def _item_found(item: str, labels: list[str]) -> bool:
    expected = _compact(item)
    return any(
        expected == label or expected in label or label in expected
        for label in labels
        if label
    )


def _expected_value_found(
    rows: list[list[str]],
    item: str,
    expected_value: str,
) -> bool:
    item_key = _compact(item)
    value_key = _compact(expected_value)
    for row in rows:
        if item_key not in _compact(_row_label(row)):
            continue
        if value_key in _compact("|".join(map(str, row))):
            return True
    return False


def evaluate_extraction(
    case: dict,
    tables: Iterable[ExtractedTable],
) -> pd.DataFrame:
    by_id = {item.table_id: item for item in tables}
    records: list[dict] = []
    for table_id, expected in case.get("tables", {}).items():
        table = by_id.get(table_id)
        rows = table.rows if table else []
        labels = [_compact(_row_label(row)) for row in rows[1:]]
        required_items = list(expected.get("required_items", []))
        item_hits = sum(_item_found(item, labels) for item in required_items)
        item_recall = (
            item_hits / len(required_items) if required_items else 1.0
        )
        expected_values = list(expected.get("expected_values", []))
        value_hits = sum(
            _expected_value_found(
                rows,
                str(item.get("item", "")),
                str(item.get("value", "")),
            )
            for item in expected_values
        )
        value_accuracy = (
            value_hits / len(expected_values) if expected_values else 1.0
        )
        records.append({
            "阶段": "表格提取",
            "目标表ID": table_id,
            "金标准页码": ",".join(map(str, expected.get("pages", []))),
            "实际页码": (
                ",".join(map(str, table.source_pages or [table.page]))
                if table else ""
            ),
            "页码召回率": None,
            "页码准确率": None,
            "是否完全一致": bool(table) and (
                sorted(table.source_pages or [table.page])
                == sorted(expected.get("pages", []))
            ),
            "项目召回率": item_recall,
            "关键数值准确率": value_accuracy,
        })
    return pd.DataFrame(records)


def evaluate_gold_case(
    case: dict,
    *,
    matches: Iterable[PageMatch] = (),
    tables: Iterable[ExtractedTable] = (),
) -> pd.DataFrame:
    frames = []
    match_list = list(matches)
    table_list = list(tables)
    if match_list:
        frames.append(evaluate_locator(case, match_list))
    if table_list:
        frames.append(evaluate_extraction(case, table_list))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
