"""Тесты, не требующие реальных ключей и обращения к внешним API."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot.dialog_manager import DialogService, DialogStore
from bot.knowledge_base import KnowledgeBase
from bot.vk_bot import VkBot, split_message
from config import BASE_DIR

KNOWLEDGE_PATH = BASE_DIR / "knowledge" / "raccoon_knowledge.md"
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"


class FakeGigaChatClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return "Еноты — замечательные звери, и я рад рассказать о них 🦝"


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = KnowledgeBase(KNOWLEDGE_PATH)

    def test_parses_sections(self) -> None:
        titles = [section.title for section in self.kb.sections]
        self.assertIn("Питание", titles)
        self.assertIn("Внешний вид", titles)
        self.assertIn("Границы базы знаний", titles)

    def test_returns_full_base_when_it_fits(self) -> None:
        context = self.kb.build_context("Чем питается енот?")
        self.assertEqual(context, self.kb.raw_text)

    def test_selects_relevant_sections_for_small_budget(self) -> None:
        context = self.kb.build_context("Чем питается енот зимой?", max_chars=2_500)
        self.assertLess(len(context), len(self.kb.raw_text))
        self.assertIn("## Питание", context)
        self.assertIn("## Границы базы знаний", context)


class SplitMessageTests(unittest.TestCase):
    def test_short_message_is_not_split(self) -> None:
        self.assertEqual(split_message("Привет, енот!"), ["Привет, енот!"])

    def test_long_message_is_split_within_limit(self) -> None:
        text = ("Это длинный абзац про енотов. " * 600).strip()
        chunks = split_message(text, limit=500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertTrue(all(chunk.strip() for chunk in chunks))


class VkPollingTests(unittest.TestCase):
    @staticmethod
    def _build_bot(conversations: list[dict], history: list[dict]):
        class FakeMessages:
            def __init__(self) -> None:
                self.history_calls: list[dict] = []

            def getConversations(self, count: int):  # noqa: N802
                return {"items": conversations}

            def getHistory(self, peer_id: int, count: int):  # noqa: N802
                self.history_calls.append({"peer_id": peer_id, "count": count})
                return {"items": history}

        messages = FakeMessages()
        bot = object.__new__(VkBot)
        bot._api = SimpleNamespace(messages=messages)  # type: ignore[attr-defined]
        bot._vk_lock = threading.Lock()  # type: ignore[attr-defined]
        bot._last_message_ids = {}  # type: ignore[attr-defined]
        bot._started_at = time.time()  # type: ignore[attr-defined]
        bot._dispatched = []  # type: ignore[attr-defined]
        bot._dispatch_message = bot._dispatched.append  # type: ignore[attr-defined]
        return bot, messages

    def test_first_poll_only_remembers_state(self) -> None:
        bot, messages = self._build_bot(
            [
                {
                    "conversation": {"peer": {"id": 1}},
                    "last_message": {"id": 10, "out": 1},
                }
            ],
            [],
        )
        bot._poll_once()

        self.assertEqual(bot._dispatched, [])
        self.assertEqual(bot._last_message_ids, {1: 10})
        self.assertEqual(messages.history_calls, [])

    def test_first_poll_answers_recent_incoming_message(self) -> None:
        now = time.time()
        bot, messages = self._build_bot(
            [
                {
                    "conversation": {"peer": {"id": 1}},
                    "last_message": {"id": 3, "out": 0, "date": now},
                }
            ],
            [
                {"id": 2, "out": 0, "date": now},
                {"id": 3, "out": 0, "date": now},
            ],
        )
        bot._started_at = now
        bot._poll_once()

        self.assertEqual(
            [message["id"] for message in bot._dispatched],
            [2, 3],
        )
        self.assertEqual(messages.history_calls, [{"peer_id": 1, "count": 20}])

    def test_only_new_incoming_messages_are_dispatched(self) -> None:
        bot, messages = self._build_bot(
            [{"conversation": {"peer": {"id": 1}}, "last_message": {"id": 12}}],
            [
                {"id": 9, "out": 0},
                {"id": 10, "out": 1},
                {"id": 11, "out": 0},
                {"id": 12, "out": 0},
            ],
        )
        bot._last_message_ids = {1: 10}
        bot._poll_once()

        self.assertEqual(bot._dispatched, [{"id": 11, "out": 0}, {"id": 12, "out": 0}])
        self.assertEqual(bot._last_message_ids, {1: 12})
        self.assertEqual(messages.history_calls, [{"peer_id": 1, "count": 20}])


class DialogStoreTests(unittest.TestCase):
    def test_history_is_trimmed_to_even_size(self) -> None:
        store = DialogStore(max_messages=4, ttl_minutes=180)
        for index in range(5):
            store.append(100, f"вопрос {index}", f"ответ {index}")

        messages = store.get(100)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "вопрос 3")
        self.assertEqual(messages[-1]["content"], "ответ 4")

    def test_reset_clears_history(self) -> None:
        store = DialogStore(max_messages=4, ttl_minutes=180)
        store.append(100, "вопрос", "ответ")
        store.reset(100)
        self.assertEqual(store.get(100), [])

    def test_expired_dialog_is_removed(self) -> None:
        store = DialogStore(max_messages=4, ttl_minutes=1)
        store.append(100, "вопрос", "ответ")
        store._dialogs[100].updated_at = time.monotonic() - 120
        self.assertEqual(store.get(100), [])


class DialogServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeGigaChatClient()
        self.kb = KnowledgeBase(KNOWLEDGE_PATH)
        self.store = DialogStore(max_messages=4, ttl_minutes=180)
        self.service = DialogService(
            client=self.client,  # type: ignore[arg-type]
            knowledge_base=self.kb,
            system_prompt_path=PROMPT_PATH,
            contact_form_url="https://example.com/form",
            store=self.store,
        )

    def test_system_prompt_contains_role_knowledge_and_contact(self) -> None:
        answer = self.service.reply(1, "Чем питается енот?")
        self.assertIn("Енот Валера", self.client.calls[0][0]["content"])
        self.assertIn("Енот-полоскун", self.client.calls[0][0]["content"])
        self.assertIn("https://example.com/form", self.client.calls[0][0]["content"])
        self.assertIn("Енот", answer)

    def test_history_is_passed_to_next_request(self) -> None:
        self.service.reply(1, "Первый вопрос")
        self.service.reply(1, "Второй вопрос")

        second_request = self.client.calls[1]
        contents = [message["content"] for message in second_request]
        self.assertIn("Первый вопрос", contents)
        self.assertIn("Второй вопрос", contents)
        self.assertEqual(second_request[-1]["role"], "user")

    def test_contact_fallback_without_url(self) -> None:
        service = DialogService(
            client=self.client,  # type: ignore[arg-type]
            knowledge_base=self.kb,
            system_prompt_path=PROMPT_PATH,
            contact_form_url=None,
            store=DialogStore(),
        )
        message = service.contact_message()
        self.assertIn("сообщения сообщества", message)

    def test_contact_message_with_url(self) -> None:
        self.assertIn("https://example.com/form", self.service.contact_message())


if __name__ == "__main__":
    unittest.main()
