from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd
import pdfplumber

from .solvency_pdf_locator import PageMatch

TABLE_SIGNATURES = {
    "SOLVENCY_MAIN": ("认可资产", "综合偿付能力充足率"),
    "LIQUIDITY_RISK": ("LCR1", "未来3个月"),
    "LIQUIDITY_MONITORING": ("综合退保率", "融资杠杆比例"),
    "OPERATING_METRICS": ("保险业务收入",),
    "THREE_YEAR_INVESTMENT_RETURN": (
        "投资收益率",
        "综合投资收益率",
    ),
    "ACTUAL_CAPITAL": ("核心一级资本", "实际资本合计"),
    "RECOGNIZED_ASSETS": ("投资资产", "认可资产合计"),
    "RECOGNIZED_LIABILITIES": ("准备金负债", "认可负债合计"),
    "MINIMUM_CAPITAL": ("量化风险最低资本", "最低资本"),
}
TABLE_EXCLUSIONS = {
    "SOLVENCY_MAIN": ("监管指标名称", "流动性覆盖率", "LCR1", "LCR2", "LCR3"),
    "OPERATING_METRICS": (
        "流动性风险监测指标",
        "流动性风险监管指标",
        "LCR1",
        "前五大产品的信息",
        "报告期内签单保费占前五位的产品",
        "签单保费占前五位的产品",
    ),
    "ACTUAL_CAPITAL": ("认可资产表", "认可负债表", "最低资本表"),
    "THREE_YEAR_INVESTMENT_RETURN": ("流动性覆盖率", "最低资本表"),
    "MINIMUM_CAPITAL": ("认可资产表", "认可负债表"),
}

TABLE_HEADERS = {
    "SOLVENCY_MAIN": (
        "项目", "指标名称", "本季度数", "本季度末数",
        "上季度可比数", "上季度末数", "基本情景下的下季度预测数",
        "下季度预测数", "下季度末预测数",
    ),
    "OPERATING_METRICS": ("指标名称", "本季度", "本年累计"),
    "ACTUAL_CAPITAL": ("行次", "期末数", "期初数"),
    "THREE_YEAR_INVESTMENT_RETURN": ("投资收益率", "近三年"),
    "MINIMUM_CAPITAL": ("行次", "期末数", "期初数"),
}

TABLE_START_MARKERS = {
    "SOLVENCY_MAIN": ("指标名称", "认可资产"),
    "OPERATING_METRICS": ("主要经营指标", "效益类指标", "规模类指标", "品质类指标", "保险业务收入"),
    "ACTUAL_CAPITAL": (
        "实际资本表",
        "实际资本明细表",
        "实际资本汇总",
        "核心资本",
        "财务报表资产总额",
        "认可资产",
        "核心一级资本",
    ),
    "THREE_YEAR_INVESTMENT_RETURN": (
        "近三年（综合）投资收益率",
        "近三年平均投资收益率",
    ),
    "MINIMUM_CAPITAL": (
        "最低资本表",
        "最低资本",
        "量化风险最低资本",
        "保险风险最低资本汇总",
        "市场风险最低资本汇总",
        "信用风险最低资本汇总",
    ),
}

TABLE_END_MARKERS = {
    "SOLVENCY_MAIN": ("监管指标名称", "流动性覆盖率", "LCR1"),
    "OPERATING_METRICS": ("流动性风险监测指标", "流动性风险监管指标", "近三年综合投资收益率", "近三年（综合）投资收益率"),
    "ACTUAL_CAPITAL": (
        "（三）认可资产",
        "(三)认可资产",
        "认可资产表",
        "认可负债表",
        "最低资本表",
    ),
    "THREE_YEAR_INVESTMENT_RETURN": ("实际资本表", "认可资产表", "最低资本表"),
    "MINIMUM_CAPITAL": (),
}



@dataclass
class UnitRecord:
    """A unit declaration found in or immediately around an extracted table."""

    scope: str
    target: str
    raw_unit: str
    normalized_unit: str
    source_page: int = 0
    source_type: str = ""
    source_text: str = ""
    confidence: str = "\u9ad8"

    def dedupe_key(self) -> tuple:
        return (
            self.scope,
            self.target,
            self.normalized_unit,
            self.source_page,
            self.source_type,
            self.source_text,
        )

    def to_dict(self, table_name: str = "") -> dict:
        return {
            "\u76ee\u6807\u8868": table_name,
            "\u7269\u7406\u9875\u7801": self.source_page,
            "\u4f5c\u7528\u8303\u56f4": self.scope,
            "\u5bf9\u8c61": self.target,
            "\u539f\u59cb\u5355\u4f4d": self.raw_unit,
            "\u89c4\u8303\u5355\u4f4d": self.normalized_unit,
            "\u6765\u6e90\u7c7b\u578b": self.source_type,
            "\u6765\u6e90\u539f\u6587": self.source_text,
            "\u7f6e\u4fe1\u5ea6": self.confidence,
        }


@dataclass
class ExtractedTable:
    table_id: str
    table_name: str
    page: int
    table_index: int
    rows: list[list[str]]
    strategy: str = ""
    quality_score: float = 0.0
    evidence: str = ""
    source_pages: list[int] = field(default_factory=list)
    unit_records: list[UnitRecord] = field(default_factory=list)
    profile_strategy_id: str = ""

    @property
    def candidate_id(self) -> str:
        pages = "-".join(map(str, self.source_pages or [self.page]))
        return (
            f"{self.table_id}:{pages}:{self.profile_strategy_id}:"
            f"{self.strategy}:{self.table_index}"
        )


    def unit_summary(self) -> str:
        if not self.unit_records:
            return "\u672a\u8bc6\u522b\u5230\u660e\u786e\u5355\u4f4d\uff0c\u8bf7\u6838\u5bf9PDF\u539f\u9875\u3002"
        grouped: dict[tuple[str, str], list[UnitRecord]] = {}
        for record in self.unit_records:
            grouped.setdefault((record.scope, record.target), []).append(record)
        parts: list[str] = []
        for (scope, target), records in grouped.items():
            units = "/".join(dict.fromkeys(item.normalized_unit for item in records))
            pages = "/".join(
                dict.fromkeys(str(item.source_page) for item in records if item.source_page)
            )
            page_note = f"\uff0c\u7b2c{pages}\u9875" if pages else ""
            target_note = f"{target}\uff1a" if target else ""
            parts.append(f"{scope}{target_note}{units}{page_note}")
        return "\uff1b".join(parts)

    def units_frame(self) -> pd.DataFrame:
        columns = [
            "\u76ee\u6807\u8868", "\u7269\u7406\u9875\u7801", "\u4f5c\u7528\u8303\u56f4", "\u5bf9\u8c61", "\u539f\u59cb\u5355\u4f4d", "\u89c4\u8303\u5355\u4f4d",
            "\u6765\u6e90\u7c7b\u578b", "\u6765\u6e90\u539f\u6587", "\u7f6e\u4fe1\u5ea6",
        ]
        return pd.DataFrame(
            [record.to_dict(self.table_name) for record in self.unit_records],
            columns=columns,
        )

    def to_frame(self, *, include_unit_footer: bool = False) -> pd.DataFrame:
        width = max((len(row) for row in self.rows), default=0)
        padded = [row + [""] * (width - len(row)) for row in self.rows]
        if include_unit_footer:
            width = max(width, 2)
            padded = [row + [""] * (width - len(row)) for row in padded]
            footer = ["\u3010\u5355\u4f4d\u5907\u6ce8\u3011", self.unit_summary()]
            padded.append(footer + [""] * (width - len(footer)))
        return pd.DataFrame(padded)



def _clean_cell(value) -> str:
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", str(value).replace("\u3000", " ")).strip()
    return value


def _clean_rows(rows: list[list]) -> list[list[str]]:
    cleaned = [[_clean_cell(cell) for cell in row] for row in rows if row]
    return [row for row in cleaned if any(cell for cell in row)]
def _repair_solvency_main(rows: list[list[str]]) -> list[list[str]]:
    repaired = [["指标名称", "本季度末数", "上季度末数", "下季度末预测数"]]
    for row in rows:
        if len(row) < 3 or "指标名称" in row[0]:
            continue
        cells = row + [""] * max(0, 7 - len(row))
        label = "".join(cells[:2]).strip()
        current = cells[2].strip()
        previous = "".join(cells[3:5]).strip()
        forecast = "".join(cells[5:7]).strip()
        if label and any((current, previous, forecast)):
            repaired.append([label, current, previous, forecast])
    return repaired


def _repair_operating(rows: list[list[str]]) -> list[list[str]]:
    header_index = next(
        (index for index, row in enumerate(rows) if row and "指标名称" in row[0]),
        None,
    )
    if header_index is None:
        return rows
    repaired = [["指标名称", "本季度（末）数", "本年累计数"]]
    for row in rows[header_index + 1:]:
        cells = [cell for cell in row if cell]
        if len(cells) >= 4:
            repaired.append([cells[0], cells[1] + cells[2], cells[3]])
        elif len(cells) >= 3:
            repaired.append(cells[:3])
    return repaired




def _row_text(row: list[str]) -> str:
    return re.sub(r"\s+", "", "".join(row))


def _slice_target_rows(table_id: str, rows: list[list[str]]) -> list[list[str]]:
    """在同页多表或文字对齐结果中切出目标表边界。"""
    if not rows:
        return []
    row_texts = [_row_text(row) for row in rows]

    start_index = 0
    if table_id == "SOLVENCY_MAIN":
        header_index = next(
            (
                index
                for index, text in enumerate(row_texts)
                if "指标名称" in text and ("本季度末数" in text or "期末数" in text)
            ),
            None,
        )
        if header_index is not None:
            start_index = header_index
        else:
            metric_index = next(
                (index for index, text in enumerate(row_texts) if "核心偿付能力充足率" in text),
                0,
            )
            start_index = max(0, metric_index - 1)
    else:
        start_markers = TABLE_START_MARKERS.get(table_id, ())
        marker_index = next(
            (
                index
                for index, text in enumerate(row_texts)
                if any(marker in text for marker in start_markers)
            ),
            0,
        )
        start_index = max(0, marker_index - 1) if marker_index else 0

    end_index = len(rows)
    end_markers = TABLE_END_MARKERS.get(table_id, ())
    for index in range(start_index + 1, len(rows)):
        if any(marker in row_texts[index] for marker in end_markers):
            end_index = index
            break
    return rows[start_index:end_index]


def _candidate_quality(table_id: str, rows: list[list[str]], strategy: str) -> tuple[float, str]:
    flat_text = _row_text([cell for row in rows for cell in row])
    required = TABLE_SIGNATURES.get(table_id, ())
    required_hits = [term for term in required if term in flat_text]
    if required and len(required_hits) != len(required):
        return -1.0, "缺少目标表核心关键词"

    exclusions = TABLE_EXCLUSIONS.get(table_id, ())
    exclusion_hits = [term for term in exclusions if term in flat_text]
    if table_id == "SOLVENCY_MAIN" and exclusion_hits:
        return -1.0, f"混入其他表：{'、'.join(exclusion_hits)}"

    headers = TABLE_HEADERS.get(table_id, ())
    header_hits = [term for term in headers if term in flat_text]
    numeric_cells = sum(
        bool(re.search(r"\d", cell))
        for row in rows
        for cell in row
        if cell
    )
    widths = [len(row) for row in rows if row]
    dominant_width_ratio = 0.0
    if widths:
        dominant_width_ratio = max(widths.count(width) for width in set(widths)) / len(widths)

    score = (
        len(required_hits) * 20
        + len(header_hits) * 4
        + min(numeric_cells, 20) * 0.3
        + dominant_width_ratio * 5
        + (3 if strategy == "线框识别" else 0)
        - len(exclusion_hits) * 12
    )
    evidence_parts = [*required_hits, *header_hits]
    evidence = "、".join(dict.fromkeys(evidence_parts)) or "结构匹配"
    return round(score, 2), evidence


def _candidate_fingerprint(rows: list[list[str]]) -> str:
    return "|".join(_row_text(row) for row in rows)


def extract_table_candidates(
    pdf_bytes: bytes,
    matches: list[PageMatch],
) -> dict[str, list[ExtractedTable]]:
    """Return ranked table candidates for every target table and PDF page."""
    grouped: dict[str, list[ExtractedTable]] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for match in matches:
            for page_number in match.pages:
                if page_number < 1 or page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                strategies = (
                    ("线框识别", page.extract_tables(table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 4,
                        "join_tolerance": 4,
                        "intersection_tolerance": 5,
                    }) or []),
                    ("文字对齐识别", page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "min_words_vertical": 2,
                        "min_words_horizontal": 1,
                    }) or []),
                )
                fingerprints: dict[str, ExtractedTable] = {}
                raw_index = 0
                for strategy, tables in strategies:
                    for rows in tables:
                        raw_index += 1
                        cleaned = _slice_target_rows(match.table_id, _clean_rows(rows))
                        if len(cleaned) < 2:
                            continue
                        signatures = TABLE_SIGNATURES.get(match.table_id, ())
                        flat_text = re.sub(r"\s+", "", "".join("".join(row) for row in cleaned))
                        if signatures and not all(signature in flat_text for signature in signatures):
                            continue
                        if match.table_id == "SOLVENCY_MAIN" and max(map(len, cleaned)) >= 6:
                            cleaned = _repair_solvency_main(cleaned)
                        elif match.table_id == "OPERATING_METRICS":
                            cleaned = _repair_operating(cleaned)
                        if len(cleaned) < 2:
                            continue
                        score, evidence = _candidate_quality(match.table_id, cleaned, strategy)
                        if score < 0:
                            continue
                        candidate = ExtractedTable(
                            table_id=match.table_id,
                            table_name=match.table_name,
                            page=page_number,
                            table_index=raw_index,
                            rows=cleaned,
                            strategy=strategy,
                            quality_score=score,
                            evidence=evidence,
                        )
                        fingerprint = _candidate_fingerprint(cleaned)
                        previous = fingerprints.get(fingerprint)
                        if previous is None or candidate.quality_score > previous.quality_score:
                            fingerprints[fingerprint] = candidate

                candidates = sorted(
                    fingerprints.values(),
                    key=lambda item: (-item.quality_score, item.strategy != "线框识别", item.table_index),
                )
                if candidates:
                    grouped[f"{match.table_id}:{page_number}"] = candidates
    return grouped


def extract_tables(pdf_bytes: bytes, matches: list[PageMatch]) -> list[ExtractedTable]:
    """Compatibility API: automatically select the highest-scoring candidate."""
    groups = extract_table_candidates(pdf_bytes, matches)
    return [candidates[0] for candidates in groups.values() if candidates]

def extracted_tables_to_workbook_bytes(tables: list[ExtractedTable]) -> bytes:
    output = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for item in tables:
            pages = item.source_pages or [item.page]
            page_label = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
            base = re.sub(r"[\\/*?:\[\]]", "", f"{item.table_name}_P{page_label}")[:31] or "Table"
            name = base
            suffix = 1
            while name in used:
                suffix += 1
                name = f"{base[:27]}_{suffix}"
            used.add(name)
            item.to_frame().to_excel(writer, sheet_name=name, index=False, header=False)
    return output.getvalue()

