"""Управление диалогами и формирование запросов к GigaChat."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from bot.gigachat_client import GigaChatClient, GigaChatError
from bot.knowledge_base import KnowledgeBase

SYSTEM_PROMPT_PLACEHOLDER = "{{KNOWLEDGE_BASE}}"
CONTACT_PLACEHOLDER = "{{CONTACT_FORM}}"


@dataclass
class _Dialog:
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


class DialogStore:
    """История диалогов в памяти с ограничением размера и времени жизни."""

    def __init__(self, max_messages: int = 12, ttl_minutes: int = 180) -> None:
        if max_messages < 2:
            raise ValueError("max_messages должен быть не меньше 2.")
        # Храним пары «вопрос — ответ», поэтому количество должно быть чётным.
        self._max_messages = max_messages - (max_messages % 2)
        self._ttl_seconds = ttl_minutes * 60
        self._dialogs: dict[int, _Dialog] = {}
        self._lock = threading.Lock()

    def _is_expired(self, dialog: _Dialog, now: float) -> bool:
        return now - dialog.updated_at > self._ttl_seconds

    def _get_unlocked(self, peer_id: int, now: float) -> _Dialog | None:
        dialog = self._dialogs.get(peer_id)
        if dialog is None:
            return None
        if self._is_expired(dialog, now):
            self._dialogs.pop(peer_id, None)
            return None
        return dialog

    def _prune_unlocked(self, now: float) -> None:
        expired = [
            peer_id
            for peer_id, dialog in self._dialogs.items()
            if self._is_expired(dialog, now)
        ]
        for peer_id in expired:
            self._dialogs.pop(peer_id, None)

    def get(self, peer_id: int) -> list[dict[str, str]]:
        now = time.monotonic()
        with self._lock:
            dialog = self._get_unlocked(peer_id, now)
            if dialog is None:
                return []
            return [dict(message) for message in dialog.messages]

    def append(self, peer_id: int, user_text: str, assistant_text: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune_unlocked(now)
            dialog = self._get_unlocked(peer_id, now)
            if dialog is None:
                dialog = _Dialog()
                self._dialogs[peer_id] = dialog

            dialog.messages.append({"role": "user", "content": user_text})
            dialog.messages.append({"role": "assistant", "content": assistant_text})
            if len(dialog.messages) > self._max_messages:
                dialog.messages = dialog.messages[-self._max_messages :]
            dialog.updated_at = now

    def reset(self, peer_id: int) -> None:
        with self._lock:
            self._dialogs.pop(peer_id, None)

    def clear(self) -> None:
        with self._lock:
            self._dialogs.clear()


class DialogService:
    """Собирает системный промпт с базой знаний и ведёт диалог с моделью."""

    def __init__(
        self,
        *,
        client: GigaChatClient,
        knowledge_base: KnowledgeBase,
        system_prompt_path: str | Path,
        contact_form_url: str | None,
        store: DialogStore,
    ) -> None:
        self._client = client
        self._knowledge_base = knowledge_base
        self._prompt_path = Path(system_prompt_path)
        self._prompt_template = self._load_prompt()
        self._contact_form_url = contact_form_url
        self._store = store

    def _load_prompt(self) -> str:
        if not self._prompt_path.is_file():
            raise FileNotFoundError(
                f"Файл системного промпта не найден: {self._prompt_path}"
            )

        text = self._prompt_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"Системный промпт пуст: {self._prompt_path}")
        if SYSTEM_PROMPT_PLACEHOLDER not in text:
            raise ValueError(
                "В системном промпте нет плейсхолдера "
                f"{SYSTEM_PROMPT_PLACEHOLDER} для базы знаний."
            )
        return text

    def _contact_value(self) -> str:
        if self._contact_form_url:
            return self._contact_form_url
        return (
            "не настроена — в этом случае предложи пользователю написать "
            "в сообщения сообщества, чтобы с ним связался менеджер"
        )

    def _build_system_prompt(self, query: str) -> str:
        knowledge = self._knowledge_base.build_context(query)
        return (
            self._prompt_template.replace(SYSTEM_PROMPT_PLACEHOLDER, knowledge)
            .replace(CONTACT_PLACEHOLDER, self._contact_value())
        )

    def reply(self, peer_id: int, user_text: str) -> str:
        """Возвращает ответ на сообщение пользователя."""

        user_text = user_text.strip()
        if not user_text:
            raise ValueError("Текст сообщения пуст.")

        history = self._store.get(peer_id)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt(user_text)}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        answer = self._client.chat(messages).strip()
        if not answer:
            raise GigaChatError("Модель вернула пустой ответ.")

        self._store.append(peer_id, user_text, answer)
        return answer

    def reset(self, peer_id: int) -> None:
        self._store.reset(peer_id)

    def contact_message(self) -> str:
        if self._contact_form_url:
            return (
                "Чтобы связаться с менеджером сообщества, заполните, "
                f"пожалуйста, форму: {self._contact_form_url}"
            )
        return (
            "Напишите, пожалуйста, в сообщения сообщества — менеджер "
            "увидит обращение и свяжется с вами."
        )

    @staticmethod
    def welcome_message() -> str:
        return (
            "Здравствуйте! Я Енот Валера, помощник сообщества «Еноты» 🦝\n\n"
            "Могу рассказать о енотах: как они выглядят, где живут, чем "
            "питаются, как растят детёнышей и впадают в зимний сон. "
            "Спрашивайте — с радостью подскажу!\n\n"
            "Команды: «помощь» — подсказки, «сброс» — начать разговор заново, "
            "«контакты» — связаться с менеджером."
        )
