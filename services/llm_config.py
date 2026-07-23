from __future__ import annotations

import re
from urllib.parse import urlparse


def _is_moonshot_url(base_url: str) -> bool:
    hostname = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    return (
        hostname in {"api.moonshot.cn", "api.moonshot.ai"}
        or hostname.endswith(".moonshot.cn")
        or hostname.endswith(".moonshot.ai")
    )


def normalize_model_id(base_url: str, model: str) -> str:
    """Return the provider's canonical API model ID for known aliases."""
    value = str(model or "").strip()
    if not _is_moonshot_url(base_url):
        return value

    candidate = re.sub(r"[\s_]+", "-", value).lower()
    if re.fullmatch(
        r"kimi-k2(?:\.\d+)?(?:-thinking|-code(?:-highspeed)?)?",
        candidate,
    ):
        return candidate
    return value


def model_request_parameters(base_url: str, model: str) -> dict[str, float]:
    """Return sampling parameters that are valid for the selected model."""
    canonical_model = normalize_model_id(base_url, model)
    if _is_moonshot_url(base_url) and canonical_model in {
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
    }:
        # Kimi K2.7 Code rejects provider or OpenAI-compatible defaults.
        # Its sampling temperature is fixed and must be sent explicitly.
        return {"temperature": 1}
    if _is_moonshot_url(base_url) and (
        canonical_model in {"kimi-k2.5", "kimi-k2.6"}
        or canonical_model.startswith("kimi-k2-thinking")
    ):
        # These models have fixed sampling values. Omitting temperature lets
        # Moonshot apply the correct value for the active thinking mode.
        return {}
    return {"temperature": 0}
