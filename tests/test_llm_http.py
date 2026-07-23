from __future__ import annotations

import unittest

import requests

from services.llm_http import post_json_with_retry


class _Response:
    ok = True
    status_code = 200
    headers = {}


class LLMHttpTests(unittest.TestCase):
    def test_retries_ssl_eof_then_returns_success(self):
        calls = 0
        delays = []

        def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise requests.exceptions.SSLError("unexpected EOF")
            return _Response()

        response = post_json_with_retry(
            post,
            "https://api.moonshot.cn/v1/chat/completions",
            headers={},
            json={},
            timeout=1,
            sleep_func=delays.append,
        )

        self.assertTrue(response.ok)
        self.assertEqual(calls, 3)
        self.assertEqual(delays, [1, 2])


if __name__ == "__main__":
    unittest.main()
