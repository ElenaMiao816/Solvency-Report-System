from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
        code="PENSION_PICC",
        standard_name="人保养老",
        company_type="养老险",
        aliases=(
            "中国人民养老保险有限责任公司",
            "PICC Pension Company Limited",
        ),
    ),
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


@lru_cache(maxsize=1)
def _known_display_names() -> tuple[str, ...]:
    """Use the existing peer master as the preferred chart-label vocabulary."""
    path = Path(__file__).resolve().parents[1] / "config" / "solvency_peer_groups.json"
    try:
        groups = json.loads(path.read_text(encoding="utf-8-sig")).get("groups", {})
        names = {
            str(name).strip()
            for members in groups.values()
            if isinstance(members, list)
            for name in members
            if str(name).strip()
        }
    except (OSError, ValueError, AttributeError):
        names = set()
    return tuple(sorted(names, key=len, reverse=True))


_LEGAL_COMPANY_SUFFIX = re.compile(r"(?:股份有限公司|有限责任公司|有限公司|股份公司)$")
_INSURANCE_LEGAL_TAIL = re.compile(
    r"(?:(?:人寿|寿险|财产|健康|养老)?保险(?:集团)?(?:股份)?|保险公司)?"
    r"(?:股份有限公司|有限责任公司|有限公司|股份公司)$"
)


def known_company_standard_name(value: Any) -> str:
    """Resolve only registered aliases and unambiguous insurer legal-name tails.

    An arbitrary legal suffix is not enough to merge entities: two unrelated
    legal companies may have the same shortened display label.
    """
    normalized = canonical_company_name(value)
    if not normalized:
        return ""
    definition = COMPANY_ALIAS_LOOKUP.get(normalized)
    if definition:
        return definition.standard_name
    for short_name in _known_display_names():
        short_key = canonical_company_name(short_name)
        if normalized == short_key:
            return short_name
        if normalized.startswith(short_key) and _INSURANCE_LEGAL_TAIL.fullmatch(
            normalized[len(short_key):]
        ):
            return short_name
    return ""


def company_display_name(value: Any) -> str:
    """Shorten a legal insurer name for charts without changing its identity."""
    original = str(value or "").strip()
    normalized = canonical_company_name(original)
    if not normalized or normalized == "行业合计":
        return original
    known_name = known_company_standard_name(original)
    if known_name:
        return known_name
    base = _LEGAL_COMPANY_SUFFIX.sub("", original)
    if base == original:
        return original
    base = base.rstrip("（( ")
    return re.sub(r"(?:保险公司|保险)$", "", base) or original


def display_company_names(frame: pd.DataFrame, column: str = "公司") -> pd.DataFrame:
    """Return a presentation copy; ambiguous short-name collisions stay distinct."""
    result = frame.copy()
    if result.empty or column not in result.columns:
        return result
    originals = [str(value).strip() for value in result[column].dropna().unique()]
    proposed = {name: company_display_name(name) for name in originals}
    owners: dict[str, set[str]] = {}
    for name, label in proposed.items():
        owners.setdefault(label, set()).add(name)
    labels = {
        name: name if len(owners[label]) > 1 else label
        for name, label in proposed.items()
    }
    result[column] = result[column].map(lambda value: labels.get(str(value).strip(), value))
    return result


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


def resolve_company_identity_from_text(
    value: Any,
    company_type_map: Mapping[str, str] | None = None,
    fallback_company_type: str = "未分类",
) -> CompanyIdentity | None:
    """Resolve a configured company when a known standard name or alias occurs in text."""
    normalized = canonical_company_name(value)
    if not normalized:
        return None

    matches = [
        (len(alias), definition)
        for alias, definition in COMPANY_ALIAS_LOOKUP.items()
        if alias and alias in normalized
    ]
    if not matches:
        return None
    _, definition = max(matches, key=lambda item: item[0])
    return resolve_company_identity(
        definition.standard_name,
        company_type_map,
        fallback_company_type=fallback_company_type,
    )


def load_peer_group_config(path: str | Path) -> tuple[dict[str, str], str]:
    """Load the auditable company-to-peer-group master mapping."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("同业分类配置中的 groups 必须是对象。")

    mapping: dict[str, str] = {}
    owners: dict[str, str] = {}
    for peer_group, company_names in groups.items():
        label = str(peer_group or "").strip()
        if not label or not isinstance(company_names, list):
            raise ValueError("同业分类配置必须使用非空分类名称和公司名称列表。")
        for company_name in company_names:
            name = str(company_name or "").strip()
            key = canonical_company_name(name)
            if not key:
                continue
            if key in mapping and mapping[key] != label:
                raise ValueError(
                    f"公司“{name}”同时属于“{owners[key]}”和“{label}”，请修正同业分类配置。"
                )
            mapping[key] = label
            owners[key] = label

    default_peer_group = str(payload.get("default_peer_group", "其他") or "其他").strip()
    return mapping, default_peer_group


def resolve_peer_group(
    value: Any,
    peer_group_map: Mapping[str, str] | None,
    fallback_peer_group: str = "其他",
) -> str:
    """Resolve a peer group by standard name, historical alias, or legal-name containment."""
    fallback = str(fallback_peer_group or "").strip()
    if not peer_group_map:
        return fallback

    normalized_map = {
        canonical_company_name(name): str(peer_group or "").strip()
        for name, peer_group in peer_group_map.items()
        if canonical_company_name(name) and str(peer_group or "").strip()
    }
    original_name = str(value or "").strip()
    identity = resolve_company_identity(original_name)
    definition = COMPANY_ALIAS_LOOKUP.get(canonical_company_name(original_name))
    candidate_names = [original_name, identity.standard_name]
    if definition:
        candidate_names.extend(definition.aliases)

    candidate_keys = [canonical_company_name(name) for name in candidate_names if name]
    for key in candidate_keys:
        if normalized_map.get(key):
            return normalized_map[key]

    matches: list[tuple[int, str]] = []
    for candidate_key in candidate_keys:
        for mapped_key, peer_group in normalized_map.items():
            if len(mapped_key) >= 4 and (
                mapped_key in candidate_key or candidate_key in mapped_key
            ):
                matches.append((len(mapped_key), peer_group))
    if not matches:
        return fallback
    longest = max(length for length, _ in matches)
    longest_groups = {group for length, group in matches if length == longest}
    return next(iter(longest_groups)) if len(longest_groups) == 1 else fallback


def resolve_company_identity(
    value: Any,
    company_type_map: Mapping[str, str] | None = None,
    fallback_company_type: str = "未分类",
) -> CompanyIdentity:
    original_name = str(value or "").strip()
    key = canonical_company_name(original_name)
    definition = COMPANY_ALIAS_LOOKUP.get(key)
    registered_name = known_company_standard_name(original_name)
    standard_name = registered_name or original_name
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
        matched_by=(
            "历史名称映射"
            if definition and key != canonical_company_name(standard_name)
            else "已登记公司名称"
            if registered_name
            else "标准名称"
        ),
    )


def reconcile_known_company_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Bring previously exported full/short names under one company identity.

    Unknown entities and their supplied codes remain untouched.
    """
    result = frame.copy()
    if result.empty or "公司" not in result.columns:
        return result
    names = result["公司"].fillna("").astype(str).str.strip()
    for name in names.unique():
        if not known_company_standard_name(name):
            continue
        identity = resolve_company_identity(name)
        mask = names.eq(name)
        result.loc[mask, "公司"] = identity.standard_name
        if "标准公司名称" in result.columns:
            result.loc[mask, "标准公司名称"] = identity.standard_name
        if "公司统一编码" in result.columns:
            result.loc[mask, "公司统一编码"] = identity.company_code
        if "原始公司名称" in result.columns:
            blank_original = result["原始公司名称"].fillna("").astype(str).str.strip().eq("")
            result.loc[mask & blank_original, "原始公司名称"] = name
    return result


def apply_company_identities(
    frame: pd.DataFrame,
    company_type_map: Mapping[str, str] | None = None,
    peer_group_map: Mapping[str, str] | None = None,
    default_peer_group: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty or "公司" not in result.columns:
        return result
    for index, row in result.iterrows():
        company = str(row.get("公司", "") or "").strip()
        company_type = str(row.get("公司类型", "") or "").strip()
        company_code = str(row.get("公司统一编码", "") or "").strip()
        if (
            company == "行业合计"
            or company_type == "行业合计"
            or company_code.startswith("INDUSTRY_")
        ):
            result.at[index, "原始公司名称"] = "行业合计"
            result.at[index, "标准公司名称"] = "行业合计"
            result.at[index, "公司"] = "行业合计"
            result.at[index, "公司统一编码"] = company_code or "INDUSTRY_TOTAL"
            result.at[index, "公司类型"] = "行业合计"
            continue
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
        if not str(row.get("同业分类", "") or "").strip() and peer_group_map:
            result.at[index, "同业分类"] = resolve_peer_group(
                original,
                peer_group_map,
                fallback_peer_group=default_peer_group,
            )
    return result
