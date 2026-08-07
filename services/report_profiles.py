from __future__ import annotations

import hashlib
import io
import json
import re
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .table_strategy_registry import (
    StrategyRegistryError,
    resolve_table_strategy,
)
from .solvency_table_boundaries import boundary_variants


PROFILE_SHEET = "报告类型"
TABLE_SHEET = "目标表"
TERM_SHEET = "定位关键词"
VARIANT_SHEET = "版式变体"
DICTIONARY_SHEET = "字段字典"
COMPLETENESS_SHEET = "完整性规则"
COMPANY_SHEET = "公司来源"
GOLD_SHEET = "Gold样本"
PROFILE_WORKBOOK_SCHEMA_VERSION = "2.0"

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
    "boundary_operating_metrics_grid",
    "boundary_respect_variant_activation",
    "boundary_require_value_for_items",
    "boundary_preserve_adjustment_after_footnote",
    "completeness_require_all_signatures",
    "completeness_allow_single_value_boundary_rows",
    "completeness_track_sections",
    "completeness_require_forecast_if_present",
}
INTEGER_TABLE_FIELDS = {
    "anchor_min_structure_hits",
    "max_pages",
    "minimum_rows",
    "continuation_min_structure_hits",
    "continuation_min_numeric_lines",
    "continuation_generic_numeric_lines",
    "exclude_page_min_hits",
}
FLOAT_TABLE_FIELDS = {
    "minimum_score",
    "continuation_min_score",
    "continuation_width_tolerance",
    "completeness_source_item_recall_ratio",
}
LIST_TABLE_FIELDS = {
    "canonical_headers",
    "boundary_actions",
    "postprocess_actions",
    "forecast_markers",
    "adjacent_table_markers",
    "section_titles",
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
    "minimum_rows",
    "include_continuation",
    "continuation_min_structure_hits",
    "continuation_min_numeric_lines",
    "continuation_min_score",
    "continuation_generic_numeric_lines",
    "allow_single_row_tail",
    "continuation_width_tolerance",
    "exclude_page_min_hits",
    "canonical_headers",
    "prompt_full_table_note",
    "prompt_single_page_note",
    "boundary_actions",
    "boundary_operating_metrics_grid",
    "boundary_respect_variant_activation",
    "boundary_require_value_for_items",
    "boundary_preserve_adjustment_after_footnote",
    "postprocess_actions",
    "completeness_require_all_signatures",
    "completeness_allow_single_value_boundary_rows",
    "completeness_track_sections",
    "completeness_source_item_recall_ratio",
    "completeness_require_forecast_if_present",
    "forecast_markers",
    "forecast_required_term",
    "adjacent_table_markers",
    "section_titles",
]
PROFILE_EXPORT_FIELDS = [
    "workbook_schema_version",
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
PROFILE_REQUIRED_FIELDS_V1 = [
    field for field in PROFILE_EXPORT_FIELDS
    if field != "workbook_schema_version"
]
VARIANT_EXPORT_FIELDS = [
    "profile_id", "table_id", "variant_id", "variant_name", "priority",
    "activation_terms", "start_items", "end_items", "required_items",
    "exclude_items", "end_item_groups", "required_item_groups",
    "scope_name", "variant_note", "exact_items_only", "enabled", "notes",
]
DICTIONARY_EXPORT_FIELDS = [
    "profile_id", "table_id", "指标编码", "指标名称", "别名", "一级模块",
    "二级模块", "标准单位", "数据类型", "核心指标", "允许期间口径",
    "enabled", "notes",
]
COMPLETENESS_EXPORT_FIELDS = [
    "profile_id", "table_id", "rule_id", "rule_type", "terms",
    "minimum_count", "severity", "enabled", "notes",
]
COMPANY_EXPORT_FIELDS = [
    "profile_id", "company_name", "company_type", "report_url", "enabled",
    "valid_from", "valid_to", "notes",
]
GOLD_EXPORT_FIELDS = [
    "profile_id", "case_id", "company", "report_period", "pdf_filename",
    "sha256", "table_id", "expected_pages", "required_items",
    "expected_row_count", "enabled", "notes",
]


class ProfileValidationError(ValueError):
    """Raised when an uploaded report profile is incomplete or inconsistent."""


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _split_terms(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value if _clean(item)]
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
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
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
    workbook_schema_version: str = "1.0"
    layout_variants: tuple[Mapping[str, Any], ...] = ()
    field_dictionary: tuple[Mapping[str, Any], ...] = ()
    completeness_rules: tuple[Mapping[str, Any], ...] = ()
    companies: tuple[Mapping[str, Any], ...] = ()
    gold_samples: tuple[Mapping[str, Any], ...] = ()

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
                "workbook_schema_version": self.workbook_schema_version,
                "layout_variants": self.layout_variants,
                "field_dictionary": self.field_dictionary,
                "completeness_rules": self.completeness_rules,
                "companies": self.companies,
                "gold_samples": self.gold_samples,
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

    def taxonomy_frame(self) -> pd.DataFrame:
        """Return an embedded v2 dictionary in the legacy taxonomy shape."""
        if not self.field_dictionary:
            return pd.DataFrame()
        columns = [
            "指标编码", "指标名称", "别名", "一级模块", "二级模块", "标准单位",
            "数据类型", "核心指标", "允许期间口径", "说明",
        ]
        rows = []
        for item in self.field_dictionary:
            row = {column: item.get(column, "") for column in columns}
            row["说明"] = item.get("notes", item.get("说明", ""))
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def company_frame(self) -> pd.DataFrame:
        """Return embedded v2 company sources using the configured column names."""
        if not self.companies:
            return pd.DataFrame()
        name_column = _clean(self.monitoring.get("company_name_column")) or "公司名称"
        type_column = _clean(self.monitoring.get("company_type_column")) or "公司类型"
        url_column = _clean(self.monitoring.get("report_url_column")) or "偿付能力报告披露网址"
        return pd.DataFrame([
            {
                name_column: item.get("company_name", ""),
                type_column: item.get("company_type", ""),
                url_column: item.get("report_url", ""),
            }
            for item in self.companies
        ])


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


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if _clean(value)))


def _split_term_groups(value: Any) -> list[list[str]]:
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple, set)) for item in value):
            return [
                [_clean(term) for term in group if _clean(term)]
                for group in value
                if any(_clean(term) for term in group)
            ]
        terms = [_clean(item) for item in value if _clean(item)]
        return [terms] if terms else []
    text = _clean(value)
    if not text:
        return []
    return [
        _split_terms(group)
        for group in re.split(r"[;；]+", text)
        if _split_terms(group)
    ]


def _compile_v2_table_inputs(
    feature_config: Mapping[str, Any],
    layout_variants: Sequence[Mapping[str, Any]],
    completeness_rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile workbook v2 rows into the runtime table configuration."""
    compiled = dict(feature_config)
    tables = [dict(item) for item in feature_config.get("tables", [])]
    by_id = {_clean(item.get("table_id")): item for item in tables}

    variants_by_table: dict[str, list[dict[str, Any]]] = {}
    for row in layout_variants:
        table_id = _clean(row.get("table_id"))
        if table_id not in by_id:
            raise ProfileValidationError(f"版式变体引用了未知目标表：{table_id}")
        variant = {
            "id": _clean(row.get("variant_id")) or f"{table_id}_VARIANT",
            "name": _clean(row.get("variant_name")) or "人工配置版式",
            "priority": int(float(row.get("priority", 0) or 0)),
            "activation_terms": _split_terms(row.get("activation_terms")),
            "title_terms": _split_terms(row.get("activation_terms")),
            "start_items": _split_terms(row.get("start_items")),
            "end_items": _split_terms(row.get("end_items")),
            "end_item_groups": _split_term_groups(
                row.get("end_item_groups")
            ),
            "required_items": _split_terms(row.get("required_items")),
            "required_item_groups": _split_term_groups(
                row.get("required_item_groups")
            ),
            "exclude_items": _split_terms(row.get("exclude_items")),
            "scope_name": _clean(row.get("scope_name")),
            "variant_note": _clean(row.get("variant_note")),
            "exact_items_only": _as_bool(row.get("exact_items_only"), False),
        }
        if not variant["end_item_groups"] and variant["end_items"]:
            variant["end_item_groups"] = [variant["end_items"]]
        if not variant["required_item_groups"] and variant["required_items"]:
            variant["required_item_groups"] = [
                [item] for item in variant["required_items"]
            ]
        variants_by_table.setdefault(table_id, []).append(variant)

    rules_by_table: dict[str, list[dict[str, Any]]] = {}
    for row in completeness_rules:
        table_id = _clean(row.get("table_id"))
        if table_id not in by_id:
            raise ProfileValidationError(f"完整性规则引用了未知目标表：{table_id}")
        rule = dict(row)
        rule["terms"] = _split_terms(row.get("terms"))
        rules_by_table.setdefault(table_id, []).append(rule)

    for table_id, table in by_id.items():
        variants = variants_by_table.get(table_id, [])
        if variants:
            table["boundary_variants"] = variants
            table["scope_name"] = next(
                (
                    variant["scope_name"]
                    for variant in variants
                    if variant.get("scope_name")
                ),
                table_id,
            )
            table["variant_note"] = "；".join(_dedupe([
                str(variant.get("variant_note", "")).strip()
                for variant in variants
            ]))
            table["profile_exclude_items"] = _dedupe([
                term for variant in variants for term in variant["exclude_items"]
            ])
        rules = rules_by_table.get(table_id, [])
        if rules:
            table["completeness_rules"] = rules
            table["completeness_terms"] = _dedupe([
                term
                for rule in rules
                if _clean(rule.get("rule_type")) == "required_terms"
                for term in rule.get("terms", [])
            ])
            minimum_rows = [
                int(float(rule.get("minimum_count", 0) or 0))
                for rule in rules
                if _clean(rule.get("rule_type")) == "minimum_rows"
            ]
            if minimum_rows:
                table["minimum_rows"] = max(minimum_rows)
    compiled["tables"] = tables
    return compiled


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
    layout_variants = tuple(dict(item) for item in payload.get("layout_variants", []))
    field_dictionary = tuple(dict(item) for item in payload.get("field_dictionary", []))
    completeness_rules = tuple(dict(item) for item in payload.get("completeness_rules", []))
    companies = tuple(dict(item) for item in payload.get("companies", []))
    gold_samples = tuple(dict(item) for item in payload.get("gold_samples", []))
    feature_config = _compile_v2_table_inputs(
        feature_config,
        layout_variants,
        completeness_rules,
    )
    contextual_tables: list[dict[str, Any]] = []
    for raw_table in feature_config.get("tables", []):
        table = dict(raw_table)
        table.setdefault("prompt_role", _clean(locator.get("prompt_role")))
        table.setdefault(
            "profile_instructions",
            list(locator.get("instructions", [])),
        )
        contextual_tables.append(table)
    feature_config = {
        **feature_config,
        "tables": contextual_tables,
    }
    feature_config = _resolve_feature_strategies(feature_config)
    monitoring_required = [
        "company_name_column", "company_type_column", "report_url_column",
    ]
    normalization_required = ["standard_template_file"]
    if not companies:
        monitoring_required.extend(["company_source_file", "company_source_sheet"])
    if not field_dictionary:
        normalization_required.append("taxonomy_file")
    required_sections = [
        (
            "monitoring",
            monitoring,
            monitoring_required,
        ),
        ("normalization", normalization, normalization_required),
        ("validation", validation, ("validation_rules_file",)),
    ]
    for section_name, section, required_fields in required_sections:
        missing = [
            field for field in required_fields
            if not _clean(section.get(field))
        ]
        if missing:
            raise ProfileValidationError(
                f"{section_name} 缺少字段：{', '.join(missing)}"
            )
    _validate_feature_config(feature_config)
    resource_fields = [
        ("normalization", "standard_template_file"),
        ("validation", "validation_rules_file"),
    ]
    if not companies:
        resource_fields.append(("monitoring", "company_source_file"))
    if not field_dictionary:
        resource_fields.append(("normalization", "taxonomy_file"))
    for section_name, field in resource_fields:
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
        workbook_schema_version=(
            _clean(payload.get("workbook_schema_version")) or "1.0"
        ),
        layout_variants=layout_variants,
        field_dictionary=field_dictionary,
        completeness_rules=completeness_rules,
        companies=companies,
        gold_samples=gold_samples,
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
        "workbook_schema_version": (
            _clean(row.get("workbook_schema_version")) or "1.0"
        ),
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


def _optional_sheet_records(
    excel: pd.ExcelFile,
    sheet_name: str,
    expected_columns: Sequence[str],
    *,
    profile_id: str,
    required_columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if sheet_name not in excel.sheet_names:
        return []
    frame = pd.read_excel(excel, sheet_name=sheet_name).fillna("")
    required = tuple(required_columns or expected_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ProfileValidationError(
            f"“{sheet_name}”缺少字段：" + "、".join(missing)
        )
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = ""
    records: list[dict[str, Any]] = []
    for row_number, row in frame.iterrows():
        record = {column: row.get(column, "") for column in expected_columns}
        row_profile_id = _clean(record.get("profile_id"))
        if not any(_clean(value) for value in record.values()):
            continue
        if row_profile_id != profile_id:
            raise ProfileValidationError(
                f"“{sheet_name}”第 {row_number + 2} 行 profile_id 应为 {profile_id}。"
            )
        if "enabled" in record and not _as_bool(record.get("enabled"), True):
            continue
        records.append(record)
    return records


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
        field for field in PROFILE_REQUIRED_FIELDS_V1
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
            elif field in LIST_TABLE_FIELDS:
                parsed = _split_terms(value)
                if parsed:
                    table[field] = parsed
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

    layout_variants = _optional_sheet_records(
        excel,
        VARIANT_SHEET,
        VARIANT_EXPORT_FIELDS,
        profile_id=profile_id,
        required_columns=(
            "profile_id", "table_id", "variant_id", "variant_name",
            "priority", "activation_terms", "start_items", "end_items",
            "required_items", "exclude_items", "exact_items_only",
            "enabled", "notes",
        ),
    )
    field_dictionary = _optional_sheet_records(
        excel, DICTIONARY_SHEET, DICTIONARY_EXPORT_FIELDS, profile_id=profile_id,
    )
    completeness_rules = _optional_sheet_records(
        excel, COMPLETENESS_SHEET, COMPLETENESS_EXPORT_FIELDS, profile_id=profile_id,
    )
    companies = _optional_sheet_records(
        excel, COMPANY_SHEET, COMPANY_EXPORT_FIELDS, profile_id=profile_id,
    )
    gold_samples = _optional_sheet_records(
        excel, GOLD_SHEET, GOLD_EXPORT_FIELDS, profile_id=profile_id,
    )
    known_table_ids = set(by_id)
    for sheet_name, records in (
        (VARIANT_SHEET, layout_variants),
        (COMPLETENESS_SHEET, completeness_rules),
        (GOLD_SHEET, gold_samples),
    ):
        for record in records:
            table_id = _clean(record.get("table_id"))
            if table_id and table_id not in known_table_ids:
                raise ProfileValidationError(
                    f"“{sheet_name}”引用了未知目标表：{table_id}"
                )
    payload.update({
        "layout_variants": layout_variants,
        "field_dictionary": field_dictionary,
        "completeness_rules": completeness_rules,
        "companies": companies,
        "gold_samples": gold_samples,
    })

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


def _export_layout_variants(profile: ReportProfile) -> list[dict[str, Any]]:
    source = list(profile.layout_variants)
    if not source:
        for table in profile.tables:
            configured = table.get("boundary_variants") or boundary_variants(
                str(table.get("table_id", ""))
            )
            for index, variant in enumerate(configured, start=1):
                end_items = list(variant.get("end_items", []))
                if not end_items:
                    end_items = [
                        term
                        for group in variant.get("end_item_groups", [])
                        for term in group
                    ]
                source.append({
                    "profile_id": profile.profile_id,
                    "table_id": table.get("table_id", ""),
                    "variant_id": variant.get("id", f"VARIANT_{index}"),
                    "variant_name": variant.get("name", f"版式{index}"),
                    "priority": variant.get("priority", 0),
                    "activation_terms": "|".join(variant.get("title_terms", [])),
                    "start_items": "|".join(variant.get("start_items", [])),
                    "end_items": "|".join(end_items),
                    "required_items": "|".join(variant.get("required_items", [])),
                    "exclude_items": "|".join(variant.get("exclude_items", [])),
                    "exact_items_only": variant.get("exact_items_only", False),
                    "enabled": True,
                    "notes": "由当前策略边界导出，可在工作簿中人工维护。",
                })
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(source, start=1):
        row = {field: raw.get(field, "") for field in VARIANT_EXPORT_FIELDS}
        row["profile_id"] = profile.profile_id
        row["variant_id"] = row["variant_id"] or f"VARIANT_{index}"
        row["enabled"] = raw.get("enabled", True)
        rows.append(row)
    return rows


def _export_field_dictionary(profile: ReportProfile) -> list[dict[str, Any]]:
    source = list(profile.field_dictionary)
    if not source and _clean(profile.normalization.get("taxonomy_file")):
        path = profile.resource_path("normalization", "taxonomy_file")
        frame = pd.read_excel(path, sheet_name="指标字典", header=2).fillna("")
        source = frame.to_dict("records")
    rows: list[dict[str, Any]] = []
    for raw in source:
        row = {field: raw.get(field, "") for field in DICTIONARY_EXPORT_FIELDS}
        row["profile_id"] = profile.profile_id
        row["table_id"] = raw.get("table_id", raw.get("目标表ID", ""))
        row["enabled"] = raw.get("enabled", True)
        row["notes"] = raw.get("notes", raw.get("说明", ""))
        rows.append(row)
    return rows


def _export_companies(profile: ReportProfile) -> list[dict[str, Any]]:
    source = list(profile.companies)
    if not source and _clean(profile.monitoring.get("company_source_file")):
        path = profile.resource_path("monitoring", "company_source_file")
        frame = pd.read_excel(
            path,
            sheet_name=_clean(profile.monitoring.get("company_source_sheet")),
            header=int(profile.monitoring.get("company_source_header", 0) or 0),
        ).fillna("")
        source = [
            {
                "company_name": row.get(profile.monitoring["company_name_column"], ""),
                "company_type": row.get(profile.monitoring["company_type_column"], ""),
                "report_url": row.get(profile.monitoring["report_url_column"], ""),
            }
            for _, row in frame.iterrows()
        ]
    rows = []
    for raw in source:
        row = {field: raw.get(field, "") for field in COMPANY_EXPORT_FIELDS}
        row["profile_id"] = profile.profile_id
        row["enabled"] = raw.get("enabled", True)
        rows.append(row)
    return rows


def _export_completeness_rules(profile: ReportProfile) -> list[dict[str, Any]]:
    source = list(profile.completeness_rules)
    if not source:
        for table in profile.tables:
            terms = table.get("completeness_terms") or table.get("content_terms", [])
            source.append({
                "table_id": table.get("table_id", ""),
                "rule_id": f"{table.get('table_id', '')}_REQUIRED_ANY",
                "rule_type": "required_any",
                "terms": "|".join(terms),
                "minimum_count": 1,
                "severity": "warning",
                "enabled": True,
                "notes": "至少命中一项代表性指标；建议用 Gold 样本继续校准。",
            })
    rows = []
    for raw in source:
        row = {field: raw.get(field, "") for field in COMPLETENESS_EXPORT_FIELDS}
        row["profile_id"] = profile.profile_id
        if isinstance(row.get("terms"), (list, tuple)):
            row["terms"] = "|".join(row["terms"])
        row["enabled"] = raw.get("enabled", True)
        rows.append(row)
    return rows


def profile_workbook_bytes(profile: ReportProfile) -> bytes:
    profile_row = {
        "workbook_schema_version": PROFILE_WORKBOOK_SCHEMA_VERSION,
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
        for field in LIST_TABLE_FIELDS:
            value = table_row.get(field, "")
            if isinstance(value, (list, tuple, set)):
                table_row[field] = "|".join(
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                )
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
    variant_rows = _export_layout_variants(profile)
    dictionary_rows = _export_field_dictionary(profile)
    completeness_rows = _export_completeness_rules(profile)
    company_rows = _export_companies(profile)
    gold_rows = []
    for raw in profile.gold_samples:
        row = {field: raw.get(field, "") for field in GOLD_EXPORT_FIELDS}
        row["profile_id"] = profile.profile_id
        row["enabled"] = raw.get("enabled", True)
        gold_rows.append(row)
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
        pd.DataFrame(variant_rows, columns=VARIANT_EXPORT_FIELDS).to_excel(
            writer, sheet_name=VARIANT_SHEET, index=False,
        )
        pd.DataFrame(dictionary_rows, columns=DICTIONARY_EXPORT_FIELDS).to_excel(
            writer, sheet_name=DICTIONARY_SHEET, index=False,
        )
        pd.DataFrame(completeness_rows, columns=COMPLETENESS_EXPORT_FIELDS).to_excel(
            writer, sheet_name=COMPLETENESS_SHEET, index=False,
        )
        pd.DataFrame(company_rows, columns=COMPANY_EXPORT_FIELDS).to_excel(
            writer, sheet_name=COMPANY_SHEET, index=False,
        )
        pd.DataFrame(gold_rows, columns=GOLD_EXPORT_FIELDS).to_excel(
            writer, sheet_name=GOLD_SHEET, index=False,
        )
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
