from __future__ import annotations

import hashlib
import io
import json
import re
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .table_strategy_registry import (
    StrategyRegistryError,
    resolve_table_strategy,
)


PROFILE_SHEET = "报告类型"
TABLE_SHEET = "目标表"
TERM_SHEET = "定位关键词"

LIST_CONFIG_FIELDS = {
    "title": "title_terms",
    "content": "content_terms",
    "header": "header_terms",
    "continuation": "continuation_terms",
    "stop": "stop_terms",
    "terminal": "terminal_terms",
    "start_item": "start_item_terms",
    "end_item": "end_item_terms",
    "exclude_item": "exclude_item_terms",
    "exclude_page": "exclude_page_terms",
}
CONFIG_FIELD_TO_RULE = {
    config_field: rule_type
    for rule_type, config_field in LIST_CONFIG_FIELDS.items()
}
BOOLEAN_TABLE_FIELDS = {
    "include_continuation",
    "allow_single_row_tail",
    "required",
}
INTEGER_TABLE_FIELDS = {
    "anchor_min_structure_hits",
    "max_pages",
    "continuation_min_structure_hits",
    "continuation_min_numeric_lines",
    "continuation_generic_numeric_lines",
    "exclude_page_min_hits",
}
FLOAT_TABLE_FIELDS = {
    "minimum_score",
    "continuation_min_score",
    "continuation_width_tolerance",
}
TABLE_EXPORT_FIELDS = [
    "table_id",
    "table_name",
    "strategy_id",
    "section_code",
    "required",
    "minimum_score",
    "anchor_min_structure_hits",
    "max_pages",
    "include_continuation",
    "continuation_min_structure_hits",
    "continuation_min_numeric_lines",
    "continuation_min_score",
    "continuation_generic_numeric_lines",
    "allow_single_row_tail",
    "continuation_width_tolerance",
    "exclude_page_min_hits",
]
PROFILE_EXPORT_FIELDS = [
    "profile_id",
    "profile_name",
    "sector",
    "report_family",
    "frequency",
    "company_types",
    "config_version",
    "locator_config_version",
    "company_source_file",
    "company_source_sheet",
    "company_source_header",
    "company_name_column",
    "company_type_column",
    "report_url_column",
    "report_terms",
    "taxonomy_file",
    "validation_rules_file",
    "standard_template_file",
    "prompt_role",
    "locator_instructions",
    "comparison_scope",
]


class ProfileValidationError(ValueError):
    """Raised when an uploaded report profile is incomplete or inconsistent."""


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _split_terms(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"[|\n\r]+", text)
        if item.strip()
    ]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, bool):
        return value
    normalized = _clean(value).lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "y", "是", "启用"}:
        return True
    if normalized in {"0", "false", "no", "n", "否", "停用"}:
        return False
    raise ProfileValidationError(f"无法识别布尔值：{value}")


def _as_number(value: Any, field: str) -> int | float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = _clean(value)
    if not text:
        return None
    try:
        if field in INTEGER_TABLE_FIELDS:
            return int(float(text))
        return float(text)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(
            f"目标表字段 {field} 必须是数字，当前值为：{value}"
        ) from exc


def _safe_project_path(project_root: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ProfileValidationError("profile 中存在空的资源文件路径。")
    root = project_root.resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProfileValidationError(
            f"profile 资源必须位于项目目录内：{raw_path}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class ReportProfile:
    profile_id: str
    profile_name: str
    sector: str
    report_family: str
    frequency: str
    company_types: tuple[str, ...]
    config_version: str
    monitoring: Mapping[str, Any]
    locator: Mapping[str, Any]
    normalization: Mapping[str, Any]
    validation: Mapping[str, Any]
    analysis: Mapping[str, Any]
    feature_config: Mapping[str, Any]
    project_root: Path
    source_name: str

    @property
    def tables(self) -> list[dict]:
        return [dict(item) for item in self.feature_config.get("tables", [])]

    @property
    def runtime_version(self) -> str:
        payload = json.dumps(
            {
                "profile_id": self.profile_id,
                "config_version": self.config_version,
                "feature_config": self.feature_config,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return f"{self.profile_id}:{hashlib.sha256(payload).hexdigest()[:12]}"

    def resource_path(self, section: str, field: str) -> Path:
        section_config = getattr(self, section)
        return _safe_project_path(
            self.project_root,
            _clean(section_config.get(field)),
        )

    def locator_context(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "prompt_role": _clean(self.locator.get("prompt_role")),
            "instructions": list(self.locator.get("instructions", [])),
        }


def _validate_feature_config(feature_config: Mapping[str, Any]) -> None:
    tables = feature_config.get("tables", [])
    if not isinstance(tables, list) or not tables:
        raise ProfileValidationError("profile 至少需要配置一张目标表。")
    seen: set[str] = set()
    for index, table in enumerate(tables, start=1):
        table_id = _clean(table.get("table_id"))
        table_name = _clean(table.get("table_name"))
        if not table_id or not table_name:
            raise ProfileValidationError(
                f"目标表第 {index} 行缺少 table_id 或 table_name。"
            )
        if table_id in seen:
            raise ProfileValidationError(f"目标表 ID 重复：{table_id}")
        seen.add(table_id)
        if not any(
            table.get(field)
            for field in ("title_terms", "content_terms", "header_terms")
        ):
            raise ProfileValidationError(
                f"目标表 {table_id} 至少需要标题、内容或表头关键词。"
            )
        max_pages = int(table.get("max_pages", 1) or 1)
        if max_pages < 1:
            raise ProfileValidationError(
                f"目标表 {table_id} 的 max_pages 必须大于等于 1。"
            )


def _resolve_feature_strategies(
    feature_config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(feature_config)
    normalized_tables: list[dict[str, Any]] = []
    for raw_table in feature_config.get("tables", []):
        table = dict(raw_table)
        table_id = _clean(table.get("table_id"))
        try:
            strategy = resolve_table_strategy(
                table_id,
                _clean(table.get("strategy_id")) or None,
            )
        except StrategyRegistryError as exc:
            raise ProfileValidationError(str(exc)) from exc
        table["strategy_id"] = strategy.strategy_id
        normalized_tables.append(table)
    normalized["tables"] = normalized_tables
    return normalized


def _build_profile(
    payload: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    *,
    project_root: Path,
    source_name: str,
) -> ReportProfile:
    profile_id = _clean(payload.get("profile_id"))
    profile_name = _clean(payload.get("profile_name"))
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", profile_id):
        raise ProfileValidationError(
            "profile_id 必须使用大写字母、数字和下划线，并以字母开头。"
        )
    if not profile_name:
        raise ProfileValidationError("profile_name 不能为空。")
    company_types = tuple(
        item
        for item in payload.get("company_types", [])
        if _clean(item)
    )
    if not company_types:
        raise ProfileValidationError("company_types 至少需要一个公司类型。")
    monitoring = dict(payload.get("monitoring", {}))
    locator = dict(payload.get("locator", {}))
    normalization = dict(payload.get("normalization", {}))
    validation = dict(payload.get("validation", {}))
    analysis = dict(payload.get("analysis", {}))
    feature_config = _resolve_feature_strategies(feature_config)
    for section_name, section, required_fields in (
        (
            "monitoring",
            monitoring,
            (
                "company_source_file",
                "company_source_sheet",
                "company_name_column",
                "company_type_column",
                "report_url_column",
            ),
        ),
        ("normalization", normalization, ("taxonomy_file", "standard_template_file")),
        ("validation", validation, ("validation_rules_file",)),
    ):
        missing = [
            field for field in required_fields
            if not _clean(section.get(field))
        ]
        if missing:
            raise ProfileValidationError(
                f"{section_name} 缺少字段：{', '.join(missing)}"
            )
    _validate_feature_config(feature_config)
    for section_name, field in (
        ("monitoring", "company_source_file"),
        ("normalization", "taxonomy_file"),
        ("normalization", "standard_template_file"),
        ("validation", "validation_rules_file"),
    ):
        resource = _safe_project_path(
            project_root,
            _clean(
                {
                    "monitoring": monitoring,
                    "normalization": normalization,
                    "validation": validation,
                }[section_name].get(field)
            ),
        )
        if not resource.exists():
            raise ProfileValidationError(
                f"profile 资源文件不存在：{resource}"
            )
    return ReportProfile(
        profile_id=profile_id,
        profile_name=profile_name,
        sector=_clean(payload.get("sector")),
        report_family=_clean(payload.get("report_family")),
        frequency=_clean(payload.get("frequency")) or "QUARTERLY",
        company_types=company_types,
        config_version=_clean(payload.get("config_version")) or "1.0",
        monitoring=monitoring,
        locator=locator,
        normalization=normalization,
        validation=validation,
        analysis=analysis,
        feature_config=dict(feature_config),
        project_root=project_root.resolve(),
        source_name=source_name,
    )


def load_profile_file(path: str | Path, project_root: str | Path) -> ReportProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    locator = dict(payload.get("locator", {}))
    feature_path = _safe_project_path(
        Path(project_root),
        _clean(locator.get("feature_config_file")),
    )
    with feature_path.open("r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    return _build_profile(
        payload,
        feature_config,
        project_root=Path(project_root),
        source_name=profile_path.name,
    )


def load_profile_registry(
    profile_dir: str | Path,
    project_root: str | Path,
) -> dict[str, ReportProfile]:
    directory = Path(profile_dir)
    profiles = [
        load_profile_file(path, project_root)
        for path in sorted(directory.glob("*.json"))
    ]
    result = {profile.profile_id: profile for profile in profiles}
    if not result:
        raise ProfileValidationError(f"未在 {directory} 找到报告 profile。")
    if len(result) != len(profiles):
        raise ProfileValidationError("报告 profile_id 存在重复。")
    return result


def _profile_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    instructions = _split_terms(row.get("locator_instructions"))
    return {
        "profile_id": _clean(row.get("profile_id")),
        "profile_name": _clean(row.get("profile_name")),
        "sector": _clean(row.get("sector")),
        "report_family": _clean(row.get("report_family")),
        "frequency": _clean(row.get("frequency")) or "QUARTERLY",
        "company_types": _split_terms(row.get("company_types")),
        "config_version": _clean(row.get("config_version")) or "1.0",
        "locator_config_version": (
            _clean(row.get("locator_config_version"))
            or _clean(row.get("config_version"))
            or "1.0"
        ),
        "monitoring": {
            "company_source_file": _clean(row.get("company_source_file")),
            "company_source_sheet": _clean(row.get("company_source_sheet")),
            "company_source_header": int(
                float(_clean(row.get("company_source_header")) or 0)
            ),
            "company_name_column": _clean(row.get("company_name_column")),
            "company_type_column": _clean(row.get("company_type_column")),
            "report_url_column": _clean(row.get("report_url_column")),
            "report_terms": _split_terms(row.get("report_terms")),
        },
        "locator": {
            "prompt_role": _clean(row.get("prompt_role")),
            "instructions": instructions,
        },
        "normalization": {
            "taxonomy_file": _clean(row.get("taxonomy_file")),
            "standard_template_file": _clean(row.get("standard_template_file")),
        },
        "validation": {
            "validation_rules_file": _clean(row.get("validation_rules_file")),
        },
        "analysis": {
            "comparison_scope": _clean(row.get("comparison_scope"))
            or "WITHIN_PROFILE",
        },
    }


def load_profile_workbook(
    workbook_bytes: bytes,
    *,
    project_root: str | Path,
    source_name: str = "uploaded_profile.xlsx",
) -> ReportProfile:
    try:
        excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    except Exception as exc:
        raise ProfileValidationError(f"无法读取 profile 工作簿：{exc}") from exc
    missing_sheets = [
        sheet
        for sheet in (PROFILE_SHEET, TABLE_SHEET, TERM_SHEET)
        if sheet not in excel.sheet_names
    ]
    if missing_sheets:
        raise ProfileValidationError(
            "profile 工作簿缺少工作表：" + "、".join(missing_sheets)
        )
    profile_frame = pd.read_excel(excel, sheet_name=PROFILE_SHEET).fillna("")
    if len(profile_frame) != 1:
        raise ProfileValidationError("“报告类型”工作表必须且只能有一行配置。")
    missing_profile_columns = [
        field for field in PROFILE_EXPORT_FIELDS
        if field not in profile_frame.columns
    ]
    if missing_profile_columns:
        raise ProfileValidationError(
            "“报告类型”缺少字段：" + "、".join(missing_profile_columns)
        )
    payload = _profile_payload_from_row(profile_frame.iloc[0].to_dict())
    profile_id = payload["profile_id"]

    table_frame = pd.read_excel(excel, sheet_name=TABLE_SHEET).fillna("")
    required_table_columns = {"table_id", "table_name"}
    if not required_table_columns.issubset(table_frame.columns):
        raise ProfileValidationError(
            "“目标表”必须包含 table_id 和 table_name。"
        )
    tables: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row_number, row in table_frame.iterrows():
        table_id = _clean(row.get("table_id"))
        if not table_id:
            continue
        table: dict[str, Any] = {
            "table_id": table_id,
            "table_name": _clean(row.get("table_name")),
        }
        for field in TABLE_EXPORT_FIELDS:
            if field in {"table_id", "table_name"} or field not in table_frame.columns:
                continue
            value = row.get(field)
            if field in BOOLEAN_TABLE_FIELDS:
                table[field] = _as_bool(
                    value,
                    default=field in {"required", "include_continuation"},
                )
            elif field in INTEGER_TABLE_FIELDS or field in FLOAT_TABLE_FIELDS:
                number = _as_number(value, field)
                if number is not None:
                    table[field] = number
            elif _clean(value):
                table[field] = _clean(value)
        if table_id in by_id:
            raise ProfileValidationError(
                f"“目标表”第 {row_number + 2} 行出现重复 ID：{table_id}"
            )
        for config_field in LIST_CONFIG_FIELDS.values():
            table[config_field] = []
        tables.append(table)
        by_id[table_id] = table

    term_frame = pd.read_excel(excel, sheet_name=TERM_SHEET).fillna("")
    required_term_columns = {"profile_id", "table_id", "rule_type", "term"}
    if not required_term_columns.issubset(term_frame.columns):
        raise ProfileValidationError(
            "“定位关键词”必须包含 profile_id、table_id、rule_type 和 term。"
        )
    for row_number, row in term_frame.iterrows():
        term_profile_id = _clean(row.get("profile_id"))
        table_id = _clean(row.get("table_id"))
        rule_type = _clean(row.get("rule_type"))
        term = _clean(row.get("term"))
        if not any((term_profile_id, table_id, rule_type, term)):
            continue
        if term_profile_id != profile_id:
            raise ProfileValidationError(
                f"“定位关键词”第 {row_number + 2} 行 profile_id "
                f"应为 {profile_id}。"
            )
        if table_id not in by_id:
            raise ProfileValidationError(
                f"“定位关键词”第 {row_number + 2} 行引用了未知目标表："
                f"{table_id}"
            )
        if rule_type not in LIST_CONFIG_FIELDS:
            raise ProfileValidationError(
                f"“定位关键词”第 {row_number + 2} 行 rule_type 无效："
                f"{rule_type}"
            )
        if "enabled" in term_frame.columns and not _as_bool(
            row.get("enabled"),
            default=True,
        ):
            continue
        if term and term not in by_id[table_id][LIST_CONFIG_FIELDS[rule_type]]:
            by_id[table_id][LIST_CONFIG_FIELDS[rule_type]].append(term)

    feature_config = {
        "version": payload["locator_config_version"],
        "description": f"{payload['profile_name']} 上传配置",
        "tables": tables,
    }
    return _build_profile(
        payload,
        feature_config,
        project_root=Path(project_root),
        source_name=source_name,
    )


def profile_workbook_bytes(profile: ReportProfile) -> bytes:
    profile_row = {
        "profile_id": profile.profile_id,
        "profile_name": profile.profile_name,
        "sector": profile.sector,
        "report_family": profile.report_family,
        "frequency": profile.frequency,
        "company_types": "|".join(profile.company_types),
        "config_version": profile.config_version,
        "locator_config_version": _clean(
            profile.feature_config.get("version")
        ) or profile.config_version,
        "company_source_file": _clean(
            profile.monitoring.get("company_source_file")
        ),
        "company_source_sheet": _clean(
            profile.monitoring.get("company_source_sheet")
        ),
        "company_source_header": int(
            profile.monitoring.get("company_source_header", 0) or 0
        ),
        "company_name_column": _clean(
            profile.monitoring.get("company_name_column")
        ),
        "company_type_column": _clean(
            profile.monitoring.get("company_type_column")
        ),
        "report_url_column": _clean(
            profile.monitoring.get("report_url_column")
        ),
        "report_terms": "|".join(profile.monitoring.get("report_terms", [])),
        "taxonomy_file": _clean(
            profile.normalization.get("taxonomy_file")
        ),
        "validation_rules_file": _clean(
            profile.validation.get("validation_rules_file")
        ),
        "standard_template_file": _clean(
            profile.normalization.get("standard_template_file")
        ),
        "prompt_role": _clean(profile.locator.get("prompt_role")),
        "locator_instructions": "\n".join(
            profile.locator.get("instructions", [])
        ),
        "comparison_scope": _clean(
            profile.analysis.get("comparison_scope")
        ) or "WITHIN_PROFILE",
    }
    table_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    for table in profile.tables:
        table_row = {
            field: table.get(field, "")
            for field in TABLE_EXPORT_FIELDS
        }
        table_row["required"] = table.get("required", True)
        table_rows.append(table_row)
        for config_field, rule_type in CONFIG_FIELD_TO_RULE.items():
            for term in table.get(config_field, []):
                term_rows.append({
                    "profile_id": profile.profile_id,
                    "table_id": table["table_id"],
                    "rule_type": rule_type,
                    "term": term,
                    "enabled": True,
                    "notes": "",
                })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(
            [profile_row],
            columns=PROFILE_EXPORT_FIELDS,
        ).to_excel(writer, sheet_name=PROFILE_SHEET, index=False)
        pd.DataFrame(
            table_rows,
            columns=TABLE_EXPORT_FIELDS,
        ).to_excel(writer, sheet_name=TABLE_SHEET, index=False)
        pd.DataFrame(
            term_rows,
            columns=[
                "profile_id",
                "table_id",
                "rule_type",
                "term",
                "enabled",
                "notes",
            ],
        ).to_excel(writer, sheet_name=TERM_SHEET, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                font = copy(cell.font)
                font.bold = True
                font.color = "FFFFFF"
                cell.font = font
                fill = copy(cell.fill)
                fill.fill_type = "solid"
                fill.fgColor.rgb = "00338D"
                cell.fill = fill
            for column_cells in sheet.columns:
                width = min(
                    42,
                    max(
                        12,
                        max(len(_clean(cell.value)) for cell in column_cells) + 2,
                    ),
                )
                sheet.column_dimensions[column_cells[0].column_letter].width = width
    return output.getvalue()
