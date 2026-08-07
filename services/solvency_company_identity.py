from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class CompanyAliasDefinition:
    code: str
    standard_name: str
    company_type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompanyIdentity:
    original_name: str
    standard_name: str
    company_code: str
    company_type: str
    matched_by: str


COMPANY_ALIASES = (
    CompanyAliasDefinition(
        code="LIFE_ORIENTAL_JIAFU",
        standard_name="东方嘉富人寿",
        company_type="寿险",
        aliases=("中韩人寿",),
    ),
)


def canonical_company_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", "", text)


def _company_code(standard_name: str) -> str:
    digest = hashlib.sha1(standard_name.encode("utf-8")).hexdigest()[:12].upper()
    return f"COMPANY_{digest}"


def _alias_lookup() -> dict[str, CompanyAliasDefinition]:
    lookup: dict[str, CompanyAliasDefinition] = {}
    for definition in COMPANY_ALIASES:
        for name in (definition.standard_name, *definition.aliases):
            lookup[canonical_company_name(name)] = definition
    return lookup


COMPANY_ALIAS_LOOKUP = _alias_lookup()


def resolve_company_identity(
    value: Any,
    company_type_map: Mapping[str, str] | None = None,
    fallback_company_type: str = "未分类",
) -> CompanyIdentity:
    original_name = str(value or "").strip()
    key = canonical_company_name(original_name)
    definition = COMPANY_ALIAS_LOOKUP.get(key)
    standard_name = definition.standard_name if definition else original_name
    company_code = definition.code if definition else _company_code(standard_name)

    normalized_type_map = {
        canonical_company_name(name): str(company_type or "").strip()
        for name, company_type in (company_type_map or {}).items()
        if canonical_company_name(name)
    }
    candidate_names = [original_name, standard_name]
    if definition:
        candidate_names.extend(definition.aliases)
    mapped_type = next(
        (
            normalized_type_map[canonical_company_name(name)]
            for name in candidate_names
            if normalized_type_map.get(canonical_company_name(name))
        ),
        "",
    )
    fallback = str(fallback_company_type or "").strip()
    company_type = (
        mapped_type
        or (fallback if fallback and fallback != "未分类" else "")
        or (definition.company_type if definition else "")
        or "未分类"
    )
    return CompanyIdentity(
        original_name=original_name,
        standard_name=standard_name,
        company_code=company_code,
        company_type=company_type,
        matched_by="历史名称映射" if definition and key != canonical_company_name(standard_name) else "标准名称",
    )


def apply_company_identities(
    frame: pd.DataFrame,
    company_type_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty or "公司" not in result.columns:
        return result
    for index, row in result.iterrows():
        original = str(row.get("原始公司名称", "") or "").strip() or row.get("公司", "")
        identity = resolve_company_identity(
            original,
            company_type_map,
            fallback_company_type=str(row.get("公司类型", "") or "").strip() or "未分类",
        )
        result.at[index, "原始公司名称"] = identity.original_name
        result.at[index, "标准公司名称"] = identity.standard_name
        result.at[index, "公司统一编码"] = identity.company_code
        result.at[index, "公司"] = identity.standard_name
        result.at[index, "公司类型"] = identity.company_type
    return result
