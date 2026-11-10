"""Чат-бот сообщества ВКонтакте на Long Poll API."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import vk_api
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.exceptions import ApiError

from bot.dialog_manager import DialogService
from bot.gigachat_client import GigaChatError

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 4_000
MAX_MESSAGE_CHARS = 3_500
FALLBACK_MESSAGE = (
    "Что-то я задумался и не смог ответить 🦝 Попробуйте, пожалуйста, "
    "ещё раз через минуту. Если вопрос срочный — напишите в сообщения "
    "сообщества, с вами свяжется менеджер."
)


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Делит длинный ответ на части по границам абзацев и предложений."""

    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = max(
            window.rfind("\n\n"),
            window.rfind("\n"),
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
            window.rfind(" "),
        )
        if cut < limit // 2:
            cut = limit

        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].strip()

    if text:
        chunks.append(text)
    return chunks


class VkBot:
    """Слушает сообщения сообщества и отвечает через DialogService."""

    def __init__(
        self,
        token: str,
        group_id: int | None,
        dialog_service: DialogService,
        transport: str = "auto",
    ) -> None:
        self._dialog_service = dialog_service
        self._vk_session = vk_api.VkApi(token=token)
        self._api = self._vk_session.get_api()
        self._group_id = group_id or self._resolve_group_id()
        self._transport = transport
        self._longpoll: VkBotLongPoll | None = None

        if transport in {"auto", "longpoll"}:
            try:
                self._longpoll = VkBotLongPoll(
                    self._vk_session,
                    group_id=self._group_id,
                    wait=25,
                )
            except Exception as exc:
                if transport == "longpoll":
                    raise RuntimeError(
                        "Не удалось запустить Long Poll API. Проверьте, что в "
                        "сообществе включены «Сообщения сообщества» и Long Poll API, "
                        "а токен имеет право «Сообщения сообщества»."
                    ) from exc
                logger.warning(
                    "Long Poll недоступен (%s). Переключаюсь на опрос сообщений.",
                    exc,
                )
                self._transport = "polling"

        if self._transport == "auto" and self._longpoll is not None:
            if not self._longpoll_events_enabled():
                logger.warning(
                    "В сообществе не включены события Long Poll (message_new). "
                    "Переключаюсь на опрос сообщений."
                )
                self._transport = "polling"

        self._last_message_ids: dict[int, int] = {}
        self._poll_interval = 2.0
        self._started_at = time.time()
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="vk-message",
        )
        self._vk_lock = threading.Lock()
        self._peer_locks: dict[int, threading.Lock] = {}
        self._peer_locks_guard = threading.Lock()
        self._seen_messages: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._running = True

    def _resolve_group_id(self) -> int:
        try:
            response = self._api.groups.getById()
        except ApiError as exc:
            raise RuntimeError(
                "Не удалось определить ID сообщества по токену. "
                "Проверьте VK_GROUP_TOKEN или укажите VK_GROUP_ID в .env."
            ) from exc

        group = response[0] if isinstance(response, list) and response else response
        if not isinstance(group, dict) or "id" not in group:
            raise RuntimeError(
                "VK API не вернул ID сообщества. Укажите VK_GROUP_ID в .env."
            )
        return int(group["id"])

    def _longpoll_events_enabled(self) -> bool:
        """Проверяет, включены ли в сообществе события message_new."""

        try:
            response = self._api.groups.getLongPollSettings(
                group_id=self._group_id
            )
        except Exception:
            # Если проверить не удалось, доверяем Long Poll.
            logger.debug("Не удалось получить настройки Long Poll.", exc_info=True)
            return True

        events = response.get("events") if isinstance(response, dict) else None
        if not isinstance(events, dict):
            return True
        return bool(events.get("message_new"))

    @staticmethod
    def check_connection(token: str, group_id: int | None = None) -> tuple[int, str]:
        """Проверяет токен и возвращает ID и название сообщества."""

        api = vk_api.VkApi(token=token).get_api()
        try:
            if group_id:
                response = api.groups.getById(group_id=group_id)
            else:
                response = api.groups.getById()
        except ApiError as exc:
            raise RuntimeError(
                "ВКонтакте отклонил запрос. Проверьте VK_GROUP_TOKEN, "
                "права ключа и VK_GROUP_ID."
            ) from exc

        group = response[0] if isinstance(response, list) and response else response
        if not isinstance(group, dict) or "id" not in group:
            raise RuntimeError("VK API не вернул данные сообщества.")
        return int(group["id"]), str(group.get("name", ""))

    def _is_duplicate(self, peer_id: int, message_id: int | None) -> bool:
        if message_id is None:
            return False

        key = (peer_id, message_id)
        if key in self._seen_messages:
            return True

        self._seen_messages[key] = None
        while len(self._seen_messages) > 2_000:
            self._seen_messages.popitem(last=False)
        return False

    def _set_typing(self, peer_id: int) -> None:
        try:
            with self._vk_lock:
                self._api.messages.setActivity(
                    type="typing",
                    peer_id=peer_id,
                    group_id=self._group_id,
                )
        except Exception:
            # Индикатор набора — необязательная функция, ошибку не показываем.
            logger.debug("Не удалось отправить статус «печатает».", exc_info=True)

    def _send(self, peer_id: int, text: str) -> None:
        chunks = split_message(text)
        if not chunks:
            chunks = [FALLBACK_MESSAGE]

        for chunk in chunks:
            for attempt in range(2):
                try:
                    with self._vk_lock:
                        self._api.messages.send(
                            peer_id=peer_id,
                            message=chunk,
                            random_id=random.getrandbits(63),
                        )
                    break
                except ApiError as exc:
                    if exc.code == 9 and attempt == 0:
                        # Flood control: ждём и пробуем ещё раз.
                        logger.warning("VK flood control, пауза перед повтором.")
                        time.sleep(1.5)
                        continue
                    if exc.code == 901:
                        logger.error(
                            "Нет доступа к сообщениям. Включите «Сообщения "
                            "сообщества» и проверьте права токена."
                        )
                    else:
                        logger.error("Ошибка VK API при отправке: %s", exc)
                    return
                except Exception:
                    logger.exception("Не удалось отправить сообщение в VK.")
                    return

        logger.info("Ответ отправлен: peer_id=%s, частей=%s", peer_id, len(chunks))

    def _get_peer_lock(self, peer_id: int) -> threading.Lock:
        with self._peer_locks_guard:
            lock = self._peer_locks.get(peer_id)
            if lock is None:
                lock = threading.Lock()
                self._peer_locks[peer_id] = lock
            return lock

    def _handle_message(self, peer_id: int, text: str) -> None:
        # Сообщения одного пользователя обрабатываем последовательно,
        # чтобы ответы не обгоняли друг друга.
        with self._get_peer_lock(peer_id):
            try:
                self._set_typing(peer_id)
                answer = self._dialog_service.reply(peer_id, text)
            except GigaChatError as exc:
                logger.warning("Ошибка GigaChat для peer_id=%s: %s", peer_id, exc)
                self._send(peer_id, FALLBACK_MESSAGE)
            except Exception:
                logger.exception("Не удалось обработать сообщение peer_id=%s", peer_id)
                self._send(peer_id, FALLBACK_MESSAGE)
            else:
                self._send(peer_id, answer)

    def _handle_command(self, peer_id: int, text: str) -> bool:
        command = text.strip().lower().split(maxsplit=1)[0] if text.strip() else ""
        # Поддержка команд с упоминанием сообщества: /start@club123.
        command = command.split("@", 1)[0]

        if command in {"/start", "start", "начать", "старт"}:
            self._send(peer_id, DialogService.welcome_message())
            return True
        if command in {"/help", "help", "помощь", "/помощь"}:
            self._send(
                peer_id,
                "Я отвечаю на вопросы о енотах 🦝\n\n"
                "Просто напишите вопрос текстом. Команда «сброс» очистит "
                "историю разговора, а «контакты» покажет, как связаться "
                "с менеджером сообщества.",
            )
            return True
        if command in {"/reset", "reset", "сброс", "/сброс", "заново"}:
            self._dialog_service.reset(peer_id)
            self._send(
                peer_id,
                "Готово! Начинаем разговор с чистого листа. "
                "О чём хотите узнать? 🦝",
            )
            return True
        if command in {"/contacts", "contacts", "контакты", "/контакты"}:
            self._send(peer_id, self._dialog_service.contact_message())
            return True
        return False

    def _on_event(self, event) -> None:
        if event.type != VkBotEventType.MESSAGE_NEW:
            return

        message = getattr(event, "message", None)
        if not isinstance(message, dict):
            obj = event.object if isinstance(event.object, dict) else {}
            message = obj.get("message")
        if not isinstance(message, dict):
            return

        self._dispatch_message(message)

    def _dispatch_message(self, message: dict) -> None:
        """Обрабатывает одно сообщение, полученное из VK."""

        if message.get("out"):
            return
        if message.get("action"):
            # Служебные события чата: добавление участника и т. п.
            return

        try:
            peer_id = int(message["peer_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Сообщение без корректного peer_id пропущено.")
            return

        message_id = message.get("id")
        if not isinstance(message_id, int):
            message_id = None
        if self._is_duplicate(peer_id, message_id):
            return

        raw_text = message.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        logger.info(
            "Входящее сообщение: peer_id=%s, символов=%s",
            peer_id,
            len(text),
        )

        if not text:
            attachments = message.get("attachments") or []
            if attachments:
                self._send(
                    peer_id,
                    "Я пока умею читать только текст 🦝 Опишите, пожалуйста, "
                    "вопрос словами — и я с радостью помогу.",
                )
            else:
                self._send(
                    peer_id,
                    "Напишите, пожалуйста, вопрос текстом — я подскажу "
                    "всё, что знаю о енотах.",
                )
            return

        if self._handle_command(peer_id, text):
            return

        if len(text) > MAX_INPUT_CHARS:
            self._send(
                peer_id,
                "Сообщение получилось очень длинным, поэтому я отвечу "
                "на первую часть. Если нужно, пришлите остальное отдельным "
                "сообщением.",
            )
            text = text[:MAX_INPUT_CHARS]

        # Ответы на сообщения одного пользователя обрабатываются по порядку,
        # а разные пользователи не блокируют друг друга.
        self._executor.submit(self._handle_message, peer_id, text)

    def _poll_once(self) -> None:
        """Забирает новые входящие сообщения через messages.getConversations."""

        with self._vk_lock:
            response = self._api.messages.getConversations(count=50)

        items = response.get("items") if isinstance(response, dict) else None
        if not isinstance(items, list):
            return

        for item in items:
            if not isinstance(item, dict):
                continue

            conversation = item.get("conversation")
            peer = conversation.get("peer") if isinstance(conversation, dict) else None
            peer_id = peer.get("id") if isinstance(peer, dict) else None
            last_message = item.get("last_message")
            last_id = (
                last_message.get("id") if isinstance(last_message, dict) else None
            )
            if not isinstance(peer_id, int) or not isinstance(last_id, int):
                continue

            previous = self._last_message_ids.get(peer_id)
            if previous is None:
                # Первый раз видим диалог. Если последнее сообщение входящее
                # и свежее, отвечаем на него, иначе просто запоминаем позицию,
                # чтобы не отвечать на старую переписку.
                if isinstance(last_message, dict) and not last_message.get("out"):
                    self._process_new_messages(
                        peer_id,
                        0,
                        last_id,
                        since=self._started_at - 5,
                    )
                self._last_message_ids[peer_id] = last_id
                continue
            if last_id <= previous:
                continue

            self._process_new_messages(peer_id, previous, last_id)
            self._last_message_ids[peer_id] = last_id

    def _process_new_messages(
        self,
        peer_id: int,
        previous: int,
        last_id: int,
        *,
        since: float | None = None,
    ) -> None:
        count = min(200, max(20, last_id - previous + 5))
        with self._vk_lock:
            response = self._api.messages.getHistory(
                peer_id=peer_id,
                count=count,
            )

        items = response.get("items") if isinstance(response, dict) else None
        if not isinstance(items, list):
            return

        incoming = [
            message
            for message in items
            if isinstance(message, dict)
            and isinstance(message.get("id"), int)
            and previous < message["id"] <= last_id
            and not message.get("out")
            and (
                since is None
                or (
                    isinstance(message.get("date"), (int, float))
                    and message["date"] >= since
                )
            )
        ]
        for message in sorted(incoming, key=lambda item: item["id"]):
            self._dispatch_message(message)

    def _shutdown(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Бот остановлен.")

    def _run_polling(self) -> None:
        logger.info(
            "Бот запущен для сообщества id=%s (опрос сообщений)",
            self._group_id,
        )
        try:
            while self._running:
                try:
                    self._poll_once()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception(
                        "Не удалось опросить сообщения. Повтор через 5 секунд."
                    )
                    time.sleep(5)
                    continue
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки.")
        finally:
            self._shutdown()

    def run(self) -> None:
        if self._transport == "polling":
            self._run_polling()
            return

        if self._longpoll is None:
            raise RuntimeError("Long Poll не инициализирован.")

        logger.info("Бот запущен для сообщества id=%s (Long Poll)", self._group_id)
        try:
            while self._running:
                try:
                    for event in self._longpoll.listen():
                        self._on_event(event)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception(
                        "Long Poll прервался. Переподключение через 5 секунд."
                    )
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки.")
        finally:
            self._shutdown()
