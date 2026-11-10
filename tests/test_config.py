"""Тесты загрузки настроек без реальных секретов."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import ConfigError, load_settings


class ConfigTests(unittest.TestCase):
    def test_required_variables_are_checked(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                load_settings()

    def test_defaults_and_even_history_size(self) -> None:
        env = {
            "VK_GROUP_TOKEN": "fake-vk-token",
            "GIGACHAT_CREDENTIALS": "ZmFrZTpmYWtl",
            "MAX_HISTORY_MESSAGES": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.gigachat_scope, "GIGACHAT_API_PERS")
        self.assertEqual(settings.gigachat_model, "GigaChat-2")
        self.assertEqual(settings.gigachat_base_url, "https://api.giga.chat/v1")
        self.assertTrue(settings.gigachat_verify_ssl)
        self.assertEqual(settings.max_history_messages, 4)
        self.assertEqual(settings.vk_transport, "auto")

    def test_transport_is_validated(self) -> None:
        env = {
            "VK_GROUP_TOKEN": "fake-vk-token",
            "GIGACHAT_CREDENTIALS": "ZmFrZTpmYWtl",
            "VK_TRANSPORT": "carrier-pigeon",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_settings()

    def test_basic_prefix_is_removed_from_credentials(self) -> None:
        env = {
            "VK_GROUP_TOKEN": "fake-vk-token",
            "GIGACHAT_CREDENTIALS": "Basic ZmFrZTpmYWtl",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.gigachat_credentials, "ZmFrZTpmYWtl")

    def test_invalid_boolean_is_rejected(self) -> None:
        env = {
            "VK_GROUP_TOKEN": "fake-vk-token",
            "GIGACHAT_CREDENTIALS": "ZmFrZTpmYWtl",
            "GIGACHAT_VERIFY_SSL": "maybe",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_settings()


if __name__ == "__main__":
    unittest.main()
