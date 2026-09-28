from __future__ import annotations

"""Blind-batch benchmark helpers for the VLM v2 pipeline."""

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .solvency_vlm_v2_pipeline import (
    VLMV2ExtractionRun,
    VLMV2LocatorRun,
    evaluate_vlm_v2_step3_gate,
)


@dataclass(frozen=True)
class VLMV2BlindGoldCase:
    case_id: str
    filename: str
    sha256: str
    pages: pd.DataFrame
    metrics: pd.DataFrame


@dataclass(frozen=True)
class VLMV2BlindScore:
    summary: pd.DataFrame
    details: pd.DataFrame


def blind_test_template_bytes() -> bytes:
    samples = pd.DataFrame(columns=["样本ID", "文件名", "SHA256", "公司", "报告期"])
    pages = pd.DataFrame(columns=["样本ID", "目标表ID", "物理页码"])
    metrics = pd.DataFrame(columns=[
        "样本ID", "目标表ID", "指标编码", "期望状态", "期望标准数值",
        "标准单位", "期间口径", "相对容差", "绝对容差",
    ])
    instructions = pd.DataFrame({"填写说明": [
        "盲测答案只在模型运行结束后参与评分，不会进入定位或抽取提示词。",
        "每份PDF在“样本”填一行；SHA256可留空，文件名需与上传PDF一致。",
        "页码金标准每个目标表填一行，跨页用英文逗号分隔，例如23,24。",
        "期望状态填写“披露数值”“横杠为零”“不适用”或“未披露”；未披露和不适用时数值可留空。",
        "期间口径填写本季度应命中的原列表头关键词；容差留空时默认相对0.5%、绝对0.01。",
    ]})
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="使用说明", index=False)
        samples.to_excel(writer, sheet_name="样本", index=False)
        pages.to_excel(writer, sheet_name="页码金标准", index=False)
        metrics.to_excel(writer, sheet_name="指标金标准", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.iter_cols():
                values = [str(cell.value or "") for cell in cells[:200]]
                worksheet.column_dimensions[cells[0].column_letter].width = min(
                    max(max((len(value) for value in values), default=0) + 2, 12), 36
                )
    return output.getvalue()


def read_blind_gold_workbook(workbook_bytes: bytes) -> dict[str, VLMV2BlindGoldCase]:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    required_sheets = {"样本", "页码金标准", "指标金标准"}
    missing = required_sheets - set(excel.sheet_names)
    if missing:
        raise ValueError("盲测金标准缺少工作表：" + "、".join(sorted(missing)))
    samples = pd.read_excel(excel, sheet_name="样本", dtype=str).fillna("")
    pages = pd.read_excel(excel, sheet_name="页码金标准", dtype=str).fillna("")
    metrics = pd.read_excel(excel, sheet_name="指标金标准").fillna("")
    for column in ("样本ID", "文件名", "SHA256"):
        if column not in samples.columns:
            raise ValueError(f"“样本”工作表缺少字段：{column}")
    for sheet_name, frame, required in (
        ("页码金标准", pages, {"样本ID", "目标表ID", "物理页码"}),
        ("指标金标准", metrics, {"样本ID", "目标表ID", "指标编码", "期望状态"}),
    ):
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"“{sheet_name}”工作表缺少字段：" + "、".join(sorted(missing_columns)))
    cases: dict[str, VLMV2BlindGoldCase] = {}
    for _, row in samples.iterrows():
        case_id = str(row["样本ID"]).strip()
        if not case_id:
            continue
        if case_id in cases:
            raise ValueError(f"样本ID重复：{case_id}")
        cases[case_id] = VLMV2BlindGoldCase(
            case_id=case_id,
            filename=str(row["文件名"]).strip(),
            sha256=str(row["SHA256"]).strip().lower(),
            pages=pages[pages["样本ID"].astype(str).str.strip() == case_id].copy(),
            metrics=metrics[metrics["样本ID"].astype(str).str.strip() == case_id].copy(),
        )
    if not cases:
        raise ValueError("盲测金标准中没有有效样本。")
    return cases


def match_blind_case(
    filename: str,
    pdf_bytes: bytes,
    cases: dict[str, VLMV2BlindGoldCase],
) -> VLMV2BlindGoldCase | None:
    digest = hashlib.sha256(pdf_bytes).hexdigest().lower()
    hash_matches = [case for case in cases.values() if case.sha256 and case.sha256 == digest]
    if len(hash_matches) == 1:
        return hash_matches[0]
    name_matches = [case for case in cases.values() if case.filename == filename]
    return name_matches[0] if len(name_matches) == 1 else None


def _pages(value: object) -> list[int]:
    values = str(value or "").replace("，", ",").split(",")
    return sorted({int(item.strip()) for item in values if item.strip().isdigit()})


def _number(value: object) -> float | None:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def score_blind_case(
    case: VLMV2BlindGoldCase,
    locator_run: VLMV2LocatorRun,
    extraction_run: VLMV2ExtractionRun,
    *,
    filename: str,
    elapsed_seconds: float,
) -> VLMV2BlindScore:
    details: list[dict[str, object]] = []
    predicted_pages = {match.table_id: sorted(set(match.pages)) for match in locator_run.matches}
    for _, gold in case.pages.iterrows():
        table_id = str(gold.get("目标表ID", "")).strip()
        expected = _pages(gold.get("物理页码", ""))
        actual = predicted_pages.get(table_id, [])
        details.append({
            "样本ID": case.case_id, "文件名": filename, "阶段": "页码定位",
            "目标表ID": table_id, "指标编码": "", "检查项": "页码完全一致",
            "期望": ",".join(map(str, expected)), "实际": ",".join(map(str, actual)),
            "通过": actual == expected,
        })

    records = extraction_run.records
    for _, gold in case.metrics.iterrows():
        table_id = str(gold.get("目标表ID", "")).strip()
        code = str(gold.get("指标编码", "")).strip()
        expected_status = str(gold.get("期望状态", "披露数值")).strip() or "披露数值"
        candidates = records[(records["目标表ID"] == table_id) & (records["指标编码"] == code)]
        row = candidates.sort_values("置信度", ascending=False).iloc[0] if not candidates.empty else None
        actual_status = str(row["状态"]) if row is not None else "not_disclosed"
        status_ok = (
            actual_status == "not_disclosed" if expected_status == "未披露"
            else actual_status == "disclosed_zero" if expected_status in {"横杠为零", "横线披露"}
            else actual_status == "disclosed_na" if expected_status == "不适用"
            else actual_status == "found"
        )
        details.append({
            "样本ID": case.case_id, "文件名": filename, "阶段": "指标抽取",
            "目标表ID": table_id, "指标编码": code, "检查项": "披露状态",
            "期望": expected_status, "实际": actual_status, "通过": status_ok,
        })
        expected_value = _number(gold.get("期望标准数值", ""))
        if expected_value is not None:
            actual_value = _number(row["标准数值"]) if row is not None else None
            rel_tol = _number(gold.get("相对容差", "")) or 0.005
            abs_tol = _number(gold.get("绝对容差", "")) or 0.01
            tolerance = max(abs_tol, abs(expected_value) * rel_tol)
            value_ok = actual_value is not None and abs(actual_value - expected_value) <= tolerance
            details.append({
                "样本ID": case.case_id, "文件名": filename, "阶段": "指标抽取",
                "目标表ID": table_id, "指标编码": code, "检查项": "标准数值",
                "期望": expected_value, "实际": actual_value if actual_value is not None else "",
                "通过": value_ok,
            })
        expected_unit = str(gold.get("标准单位", "")).strip()
        if expected_unit:
            actual_unit = str(row["标准单位"]).strip() if row is not None else ""
            details.append({
                "样本ID": case.case_id, "文件名": filename, "阶段": "指标抽取",
                "目标表ID": table_id, "指标编码": code, "检查项": "标准单位",
                "期望": expected_unit, "实际": actual_unit, "通过": actual_unit == expected_unit,
            })
        expected_period = str(gold.get("期间口径", "")).strip()
        if expected_period:
            actual_period = str(row["期间口径"]).strip() if row is not None else ""
            details.append({
                "样本ID": case.case_id, "文件名": filename, "阶段": "指标抽取",
                "目标表ID": table_id, "指标编码": code, "检查项": "本季度期间口径",
                "期望": expected_period, "实际": actual_period,
                "通过": expected_period in actual_period,
            })

    detail_frame = pd.DataFrame(details)
    page_checks = detail_frame[detail_frame["阶段"] == "页码定位"]
    status_checks = detail_frame[detail_frame["检查项"] == "披露状态"]
    value_checks = detail_frame[detail_frame["检查项"] == "标准数值"]
    period_checks = detail_frame[detail_frame["检查项"] == "本季度期间口径"]
    gate = evaluate_vlm_v2_step3_gate(extraction_run)

    def accuracy(frame: pd.DataFrame) -> float:
        return float(frame["通过"].mean()) if not frame.empty else 1.0

    summary = pd.DataFrame([{
        "样本ID": case.case_id,
        "文件名": filename,
        "页码完全准确率": accuracy(page_checks),
        "披露状态准确率": accuracy(status_checks),
        "数值准确率": accuracy(value_checks),
        "本季度口径准确率": accuracy(period_checks),
        "总体检查准确率": accuracy(detail_frame),
        "STEP3门槛": "通过" if gate.passed else "失败",
        "模型调用数": locator_run.model_calls + extraction_run.model_calls,
        "重试调用数": extraction_run.retry_calls,
        "耗时秒": round(elapsed_seconds, 2),
    }])
    return VLMV2BlindScore(summary=summary, details=detail_frame)


def combine_blind_scores(scores: Sequence[VLMV2BlindScore]) -> VLMV2BlindScore:
    if not scores:
        return VLMV2BlindScore(pd.DataFrame(), pd.DataFrame())
    summary = pd.concat([score.summary for score in scores], ignore_index=True)
    details = pd.concat([score.details for score in scores], ignore_index=True)
    accuracy_columns = [
        "页码完全准确率", "披露状态准确率", "数值准确率", "本季度口径准确率", "总体检查准确率",
    ]
    aggregate = {
        "样本ID": "批量汇总",
        "文件名": f"{len(summary)}份PDF",
        **{column: float(summary[column].mean()) for column in accuracy_columns},
        "STEP3门槛": "通过" if bool((summary["STEP3门槛"] == "通过").all()) else "失败",
        "模型调用数": int(summary["模型调用数"].sum()),
        "重试调用数": int(summary["重试调用数"].sum()),
        "耗时秒": round(float(summary["耗时秒"].sum()), 2),
    }
    summary = pd.concat([pd.DataFrame([aggregate]), summary], ignore_index=True)
    return VLMV2BlindScore(summary=summary, details=details)


def blind_scores_workbook_bytes(score: VLMV2BlindScore) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        score.summary.to_excel(writer, sheet_name="样本评分", index=False)
        score.details.to_excel(writer, sheet_name="逐项明细", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
    return output.getvalue()
