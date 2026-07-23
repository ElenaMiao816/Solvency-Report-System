from __future__ import annotations

import unittest

from services.llm_config import model_request_parameters, normalize_model_id
from services.solvency_hybrid_pipeline import _call_chat


class _Response:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "{}"}}]}


class LLMConfigTests(unittest.TestCase):
    def test_normalizes_moonshot_display_name(self):
        self.assertEqual(
            normalize_model_id("https://api.moonshot.cn/v1", "Kimi K2.6"),
            "kimi-k2.6",
        )
        self.assertEqual(
            normalize_model_id(
                "https://api.moonshot.cn/v1",
                "Kimi K2.7 Code",
            ),
            "kimi-k2.7-code",
        )

    def test_omits_temperature_for_moonshot_kimi_k2_6(self):
        self.assertEqual(
            model_request_parameters("https://api.moonshot.cn/v1", "kimi-k2.6"),
            {},
        )

    def test_uses_locked_temperature_for_moonshot_kimi_k2_7_code(self):
        self.assertEqual(
            model_request_parameters(
                "https://api.moonshot.cn/v1",
                "kimi-k2.7-code",
            ),
            {"temperature": 1},
        )

    def test_keeps_temperature_for_other_models(self):
        self.assertEqual(
            model_request_parameters(
                "https://api.deepseek.com/v1",
                "deepseek-chat",
            ),
            {"temperature": 0},
        )

    def test_chat_request_uses_valid_kimi_parameters(self):
        captured = {}

        def post(*args, **kwargs):
            captured.update(kwargs["json"])
            return _Response()

        _call_chat(
            api_key="test",
            base_url="https://api.moonshot.cn/v1",
            model="Kimi K2.6",
            messages=[{"role": "user", "content": "test"}],
            timeout=1,
            post_func=post,
        )

        self.assertEqual(captured["model"], "kimi-k2.6")
        self.assertNotIn("temperature", captured)

    def test_chat_request_uses_valid_kimi_k2_7_code_parameters(self):
        captured = {}

        def post(*args, **kwargs):
            captured.update(kwargs["json"])
            return _Response()

        _call_chat(
            api_key="test",
            base_url="https://api.moonshot.cn/v1",
            model="Kimi K2.7 Code",
            messages=[{"role": "user", "content": "test"}],
            timeout=1,
            post_func=post,
        )

        self.assertEqual(captured["model"], "kimi-k2.7-code")
        self.assertEqual(captured["temperature"], 1)


if __name__ == "__main__":
    unittest.main()
