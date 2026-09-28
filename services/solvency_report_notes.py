from __future__ import annotations

import io
from typing import Iterable

import pandas as pd

from .solvency_navigation import COMPANY_NAVIGATION, INDUSTRY_NAVIGATION, OVERVIEW_LEVEL


NOTE_COLUMNS = [
    "模块ID",
    "一级分类",
    "二级分类",
    "对应图表名称",
    "分析内容-默认",
    "分析内容-自定义",
    "注释内容",
    "图片文件名",
]

COMPANY_CHART_NAME_ALIASES = {
    "核心资本明细占比-待定": "核心一级资本明细",
    "附属资本明细占比-待定": "附属一级资本明细",
    "核心及综合充足率组合图": "核心及综合充足率",
    "综合偿付能力充足率": "核心及综合充足率",
    "核心偿付能力充足率": "核心及综合充足率",
    "综合充足率柱状图": "核心及综合充足率",
    "核心充足率柱状图": "核心及综合充足率",
    "核心资本": "核心资本占比",
    "核心资本/注册资本率": "核心资本/注册资本",
    "注册资本/核心资本率": "核心资本/注册资本",
    "利率与权益价格风险占认可资产率散点图": "利率与权益价格风险占认可资产率气泡图",
    "利差与对手违约风险占认可资产率散点图": "利差与对手违约风险占认可资产率气泡图",
    "寿险与非寿险保险风险占认可负债率散点图": "寿险与非寿险保险风险占认可负债率气泡图",
    "四级资本规模与结构": "资本规模与结构",
    "资本分级行业分布": "资本规模与结构",
    "资本结构占比": "资本规模与结构",
    "风险分散效应和损失吸收": "量化风险最低资本构成",
}
SPLIT_COMPANY_CHART_ALIASES: dict[str, tuple[str, ...]] = {}

INDUSTRY_SPECIAL_MODULES = (
    ("S8_OVERVIEW", OVERVIEW_LEVEL, "全部", "行业整体偿付能力概览"),
    ("S8_FINANCING", OVERVIEW_LEVEL, "全部", "重大融资信息统计"),
)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def clean_note_text(value: object) -> str:
    return _clean_text(value)


def _navigation_template(
    entries: Iterable[object],
    *,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        chart_name = _clean_text(getattr(entry, "chart_name", ""))
        if not chart_name or chart_name in seen:
            continue
        seen.add(chart_name)
        rows.append({
            "模块ID": f"{prefix}_{len(rows) + 1:02d}",
            "一级分类": _clean_text(getattr(entry, "level_one", "")),
            "二级分类": _clean_text(getattr(entry, "level_two", "")) or "全部",
            "对应图表名称": chart_name,
            "分析内容-默认": "",
            "分析内容-自定义": "",
            "注释内容": "",
            "图片文件名": "",
        })
    return pd.DataFrame(rows, columns=NOTE_COLUMNS)


def company_notes_template() -> pd.DataFrame:
    return _navigation_template(COMPANY_NAVIGATION, prefix="S7")


def industry_notes_template() -> pd.DataFrame:
    rows = [
        {
            "模块ID": module_id,
            "一级分类": level_one,
            "二级分类": level_two,
            "对应图表名称": chart_name,
            "分析内容-默认": "",
            "分析内容-自定义": "",
            "注释内容": "",
            "图片文件名": "",
        }
        for module_id, level_one, level_two, chart_name in INDUSTRY_SPECIAL_MODULES
    ]
    navigation = _navigation_template(
        INDUSTRY_NAVIGATION,
        prefix="S8_CHART",
    )
    return pd.concat(
        [pd.DataFrame(rows, columns=NOTE_COLUMNS), navigation],
        ignore_index=True,
    )


def notes_workbook_bytes(frame: pd.DataFrame, sheet_name: str = "分析注释") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.reindex(columns=NOTE_COLUMNS).to_excel(
            writer,
            sheet_name=sheet_name[:31],
            index=False,
        )
    return output.getvalue()


def normalize_notes_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "一级模块": "一级分类",
        "二级模块": "二级分类",
        "具体图表": "对应图表名称",
        "图表名称": "对应图表名称",
        "分析内容": "分析内容-自定义",
        "注释": "注释内容",
    }
    result = frame.copy()
    result.columns = [aliases.get(str(column).strip(), str(column).strip()) for column in result.columns]
    for column in NOTE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].map(_clean_text)
    result.loc[result["二级分类"].eq(""), "二级分类"] = "全部"
    result = result[result["对应图表名称"].ne("")].copy()
    return (
        result.reindex(columns=NOTE_COLUMNS)
        .drop_duplicates(subset=["对应图表名称"], keep="last")
        .reset_index(drop=True)
    )


def read_notes_workbook(workbook_bytes: bytes, source_name: str = "") -> pd.DataFrame:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    for sheet_name in excel.sheet_names:
        preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=12)
        for index, row in preview.iterrows():
            values = {str(value).strip() for value in row if not pd.isna(value)}
            has_chart = bool({"对应图表名称", "具体图表", "图表名称"} & values)
            if "模块ID" in values and has_chart:
                frame = pd.read_excel(excel, sheet_name=sheet_name, header=int(index))
                result = normalize_notes_frame(frame)
                if result.empty:
                    raise ValueError("分析注释表没有可识别的图表记录。")
                return result
    label = f"“{source_name}”" if source_name else "该工作簿"
    raise ValueError(f"{label}未找到包含“模块ID”和“对应图表名称”的表头。")


def overlay_notes(template: pd.DataFrame, uploaded: pd.DataFrame | None) -> pd.DataFrame:
    base = normalize_notes_frame(template)
    if not isinstance(uploaded, pd.DataFrame) or uploaded.empty:
        return base
    custom = normalize_notes_frame(uploaded)
    custom["对应图表名称"] = custom["对应图表名称"].replace(COMPANY_CHART_NAME_ALIASES)
    expanded_rows: list[dict[str, str]] = []
    for _, row in custom.iterrows():
        names = SPLIT_COMPANY_CHART_ALIASES.get(
            row["对应图表名称"],
            (row["对应图表名称"],),
        )
        for name in names:
            copied = row.to_dict()
            copied["对应图表名称"] = name
            expanded_rows.append(copied)
    custom = pd.DataFrame(expanded_rows, columns=NOTE_COLUMNS)
    custom = custom.drop_duplicates(subset=["对应图表名称"], keep="last")
    rows = {row["对应图表名称"]: row.to_dict() for _, row in base.iterrows()}
    order = base["对应图表名称"].tolist()
    for _, row in custom.iterrows():
        chart_name = row["对应图表名称"]
        if chart_name not in rows:
            order.append(chart_name)
            rows[chart_name] = {column: "" for column in NOTE_COLUMNS}
        for column in NOTE_COLUMNS:
            value = row[column]
            if value or column in {"分析内容-默认", "分析内容-自定义", "注释内容", "图片文件名"}:
                rows[chart_name][column] = value
    return pd.DataFrame([rows[name] for name in order], columns=NOTE_COLUMNS)


def notes_lookup(frame: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    normalized = normalize_notes_frame(frame)
    return {
        row["对应图表名称"]: row.to_dict()
        for _, row in normalized.iterrows()
    }
