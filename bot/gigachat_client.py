"""Минимальный клиент GigaChat API: OAuth-токен и Chat Completions."""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from typing import Any

import requests
from requests import Response, Session
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class GigaChatError(RuntimeError):
    """Ошибка при обращении к GigaChat API."""


class GigaChatClient:
    """Клиент официального GigaChat API.

    Токен доступа получается по Authorization Key (Base64 от Client ID:Secret)
    и кэшируется до истечения срока действия.
    """

    def __init__(
        self,
        credentials: str,
        *,
        scope: str,
        model: str,
        base_url: str,
        oauth_url: str,
        verify_ssl: bool = True,
        ca_bundle: str | None = None,
        temperature: float = 0.55,
        max_tokens: int = 800,
        timeout: int = 90,
        session: Session | None = None,
    ) -> None:
        self._credentials = credentials.strip()
        self._scope = scope
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._oauth_url = oauth_url
        self._verify_ssl = verify_ssl
        self._ca_bundle = ca_bundle
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "RaccoonVKBot/1.0 (community assistant)"}
        )

        self._token: str | None = None
        self._token_valid_until = 0.0
        self._token_lock = threading.Lock()

    @property
    def _verify(self) -> bool | str:
        return self._ca_bundle if self._ca_bundle else self._verify_ssl

    @staticmethod
    def _short_response_text(response: Response, limit: int = 300) -> str:
        text = (response.text or "").replace("\n", " ").strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    def _get_access_token(self, force_refresh: bool = False) -> str:
        with self._token_lock:
            now = time.time()
            if (
                not force_refresh
                and self._token
                and now < self._token_valid_until
            ):
                return self._token

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self._credentials}",
            }

            try:
                response = self._session.post(
                    self._oauth_url,
                    headers=headers,
                    data={"scope": self._scope},
                    timeout=30,
                    verify=self._verify,
                )
            except RequestException as exc:
                raise GigaChatError(
                    "Не удалось подключиться к OAuth-сервису GigaChat. "
                    "Проверьте интернет и настройки TLS."
                ) from exc

            if response.status_code != 200:
                raise GigaChatError(
                    "GigaChat не выдал access token "
                    f"(HTTP {response.status_code}): "
                    f"{self._short_response_text(response)}. "
                    "Проверьте GIGACHAT_CREDENTIALS и GIGACHAT_SCOPE."
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise GigaChatError(
                    "OAuth-сервис GigaChat вернул ответ не в формате JSON."
                ) from exc

            token = payload.get("access_token")
            if not isinstance(token, str) or not token.strip():
                raise GigaChatError(
                    "В ответе OAuth-сервиса GigaChat нет поля access_token."
                )

            expires_at = payload.get("expires_at")
            if isinstance(expires_at, (int, float)):
                expires_at_ts = float(expires_at)
                # API может вернуть время в секундах или миллисекундах.
                if expires_at_ts > 10_000_000_000:
                    expires_at_ts /= 1000
            else:
                expires_at_ts = now + 25 * 60

            self._token = token.strip()
            self._token_valid_until = max(now + 30, expires_at_ts - 60)
            logger.debug(
                "Получен новый access token GigaChat, действует ещё %.0f сек.",
                max(0, self._token_valid_until - now),
            )
            return self._token

    def _extract_content(self, payload: dict[str, Any]) -> str:
        # На случай обёртки вида {"value": {...}} в отдельных версиях API.
        if isinstance(payload.get("value"), dict):
            payload = payload["value"]

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GigaChatError("GigaChat вернул ответ без choices.")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise GigaChatError("GigaChat вернул ответ без message.")

        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            text = "\n".join(parts).strip()
        else:
            text = ""

        if not text:
            raise GigaChatError("GigaChat вернул пустой ответ.")
        return text

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Отправляет историю диалога и возвращает текст ответа модели."""

        if not messages:
            raise ValueError("Список messages не может быть пустым.")

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

        force_refresh = False
        last_error = "неизвестная ошибка"

        for attempt in range(3):
            token = self._get_access_token(force_refresh=force_refresh)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }

            try:
                response = self._session.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                    verify=self._verify,
                )
            except RequestException as exc:
                last_error = f"сетевая ошибка: {exc.__class__.__name__}"
                if attempt < 2:
                    time.sleep(2**attempt + random.uniform(0, 0.5))
                    continue
                raise GigaChatError(
                    "Не удалось связаться с GigaChat API. "
                    "Проверьте интернет и доступность api.giga.chat."
                ) from exc

            if response.status_code == 401 and attempt == 0:
                logger.warning("GigaChat отклонил access token, обновляю его.")
                force_refresh = True
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{self._short_response_text(response)}"
                )
                if attempt < 2:
                    delay = 2**attempt + random.uniform(0, 0.5)
                    logger.warning(
                        "GigaChat временно недоступен (%s), повтор через %.1f сек.",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise GigaChatError(
                    "GigaChat временно недоступен. Попробуйте позже."
                )

            if response.status_code != 200:
                raise GigaChatError(
                    f"GigaChat вернул ошибку HTTP {response.status_code}: "
                    f"{self._short_response_text(response)}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise GigaChatError(
                    "GigaChat вернул ответ не в формате JSON."
                ) from exc

            if not isinstance(data, dict):
                raise GigaChatError("GigaChat вернул неожиданный формат ответа.")

            return self._extract_content(data)

        raise GigaChatError(f"Не удалось получить ответ GigaChat: {last_error}")

    def check_connection(self) -> list[str]:
        """Проверяет OAuth и доступность API, возвращает список моделей."""

        token = self._get_access_token(force_refresh=True)
        try:
            response = self._session.get(
                f"{self._base_url}/models",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout=30,
                verify=self._verify,
            )
        except RequestException as exc:
            raise GigaChatError(
                "Не удалось получить список моделей GigaChat."
            ) from exc

        if response.status_code != 200:
            raise GigaChatError(
                f"GigaChat /models вернул HTTP {response.status_code}: "
                f"{self._short_response_text(response)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise GigaChatError(
                "GigaChat /models вернул ответ не в формате JSON."
            ) from exc

        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []

        result: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                result.append(item["id"])
        return result

    def close(self) -> None:
        self._session.close()
