"""Тесты клиента GigaChat с подменённой HTTP-сессией."""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

from bot.gigachat_client import GigaChatClient, GigaChatError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False) if payload else ""

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        if not self.responses:
            raise AssertionError("Неожиданный HTTP-запрос")
        return self.responses.pop(0)

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if not self.responses:
            raise AssertionError("Неожиданный HTTP-запрос")
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def fake_oauth(token: str = "token-1") -> FakeResponse:
    return FakeResponse(
        200,
        {"access_token": token, "expires_at": time.time() + 3_600},
    )


def fake_chat(text: str = "Привет! Я Енот Валера.") -> FakeResponse:
    return FakeResponse(
        200,
        {"choices": [{"message": {"role": "assistant", "content": text}}]},
    )


def make_client(session: FakeSession, **overrides) -> GigaChatClient:
    params = {
        "credentials": "fake-credentials",
        "scope": "GIGACHAT_API_PERS",
        "model": "GigaChat-2",
        "base_url": "https://api.example.test/v1",
        "oauth_url": "https://oauth.example.test/api/v2/oauth",
        "timeout": 5,
        "session": session,
    }
    params.update(overrides)
    return GigaChatClient(**params)  # type: ignore[arg-type]


class GigaChatClientTests(unittest.TestCase):
    def test_token_is_cached_between_requests(self) -> None:
        session = FakeSession([fake_oauth(), fake_chat("Ответ"), fake_chat("Ещё ответ")])
        client = make_client(session)

        self.assertEqual(client.chat([{"role": "user", "content": "Привет"}]), "Ответ")
        self.assertEqual(client.chat([{"role": "user", "content": "Ещё"}]), "Ещё ответ")

        oauth_calls = [call for call in session.calls if "oauth" in call[1]]
        self.assertEqual(len(oauth_calls), 1)

        chat_calls = [call for call in session.calls if "chat/completions" in call[1]]
        self.assertEqual(len(chat_calls), 2)
        self.assertEqual(
            chat_calls[0][2]["headers"]["Authorization"], "Bearer token-1"
        )

    def test_token_is_refreshed_after_401(self) -> None:
        session = FakeSession(
            [
                fake_oauth("token-old"),
                FakeResponse(401, {"message": "expired"}),
                fake_oauth("token-new"),
                fake_chat("Ответ после обновления"),
            ]
        )
        client = make_client(session)

        answer = client.chat([{"role": "user", "content": "Привет"}])
        self.assertEqual(answer, "Ответ после обновления")

        chat_calls = [call for call in session.calls if "chat/completions" in call[1]]
        self.assertEqual(
            chat_calls[-1][2]["headers"]["Authorization"], "Bearer token-new"
        )

    def test_temporary_server_error_is_retried(self) -> None:
        session = FakeSession(
            [
                fake_oauth(),
                FakeResponse(500, {"message": "temporary"}),
                fake_chat("Ответ после повтора"),
            ]
        )
        client = make_client(session)

        with patch("bot.gigachat_client.time.sleep") as sleep:
            answer = client.chat([{"role": "user", "content": "Привет"}])

        self.assertEqual(answer, "Ответ после повтора")
        sleep.assert_called()

    def test_server_error_raises_after_retries(self) -> None:
        session = FakeSession(
            [
                fake_oauth(),
                FakeResponse(500, {"message": "temporary"}),
                FakeResponse(500, {"message": "temporary"}),
                FakeResponse(500, {"message": "temporary"}),
            ]
        )
        client = make_client(session)

        with patch("bot.gigachat_client.time.sleep"):
            with self.assertRaises(GigaChatError):
                client.chat([{"role": "user", "content": "Привет"}])

    def test_supports_wrapped_response(self) -> None:
        session = FakeSession(
            [
                fake_oauth(),
                FakeResponse(
                    200,
                    {
                        "value": {
                            "choices": [
                                {"message": {"role": "assistant", "content": "Из value"}}
                            ]
                        }
                    },
                ),
            ]
        )
        client = make_client(session)
        self.assertEqual(
            client.chat([{"role": "user", "content": "Привет"}]),
            "Из value",
        )

    def test_check_connection_returns_models(self) -> None:
        session = FakeSession(
            [
                fake_oauth(),
                FakeResponse(
                    200,
                    {
                        "data": [
                            {"id": "GigaChat-2"},
                            {"id": "GigaChat-2-Pro"},
                        ]
                    },
                ),
            ]
        )
        client = make_client(session)
        self.assertEqual(
            client.check_connection(),
            ["GigaChat-2", "GigaChat-2-Pro"],
        )

    def test_empty_credentials_are_rejected_by_api(self) -> None:
        session = FakeSession([FakeResponse(401, {"message": "unauthorized"})])
        client = make_client(session)
        with self.assertRaises(GigaChatError):
            client.chat([{"role": "user", "content": "Привет"}])


if __name__ == "__main__":
    unittest.main()
