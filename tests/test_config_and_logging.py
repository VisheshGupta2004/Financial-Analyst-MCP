import io
import json
import logging
import os
import unittest
from unittest import mock

from backend.config import load_config
from backend.logging_config import (
    _JsonFormatter,
    get_request_id,
    new_request_id,
    set_request_id,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.llm.provider, "mock")
            self.assertEqual(cfg.backend.port, 8000)
            self.assertTrue(cfg.cache.enabled)

    def test_reads_overrides_from_env(self):
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o",
            "BACKEND_PORT": "9001",
            "CRITIC_ENABLED": "false",
            "CORS_ORIGINS": "http://a.com, http://b.com",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.llm.provider, "openai")
            self.assertEqual(cfg.llm.model, "gpt-4o")
            self.assertEqual(cfg.backend.port, 9001)
            self.assertFalse(cfg.critic.enabled)
            self.assertEqual(
                cfg.backend.cors_origins_list(), ["http://a.com", "http://b.com"]
            )

    def test_invalid_int_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"BACKEND_PORT": "not-a-number"}, clear=True):
            cfg = load_config()
            self.assertEqual(cfg.backend.port, 8000)

    def test_mcp_server_args_list_splits_correctly(self):
        with mock.patch.dict(
            os.environ, {"MCP_SERVER_ARGS": "-m backend.mcp_server"}, clear=True
        ):
            cfg = load_config()
            self.assertEqual(cfg.mcp.server_args_list(), ["-m", "backend.mcp_server"])


class RequestIdTests(unittest.TestCase):
    def test_new_request_id_is_reasonably_unique(self):
        ids = {new_request_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_set_and_get_round_trip(self):
        set_request_id("abc123")
        self.assertEqual(get_request_id(), "abc123")

    def test_get_before_set_returns_placeholder(self):
        # A fresh context var default; note this can be polluted by test
        # execution order within the same interpreter, so we only assert it
        # is a non-empty string, not the literal default.
        self.assertIsInstance(get_request_id(), str)
        self.assertTrue(len(get_request_id()) > 0)


class JsonFormatterTests(unittest.TestCase):
    def test_formats_valid_json_with_expected_keys(self):
        set_request_id("req-42")
        logger = logging.getLogger("test.json.formatter")
        logger.setLevel(logging.INFO)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())

        class _Filter(logging.Filter):
            def filter(self, record):
                record.request_id = get_request_id()
                return True

        handler.addFilter(_Filter())
        logger.addHandler(handler)
        logger.propagate = False

        logger.info("hello world")
        handler.flush()

        payload = json.loads(stream.getvalue().strip())
        self.assertEqual(payload["message"], "hello world")
        self.assertEqual(payload["request_id"], "req-42")
        self.assertEqual(payload["level"], "INFO")
        logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
