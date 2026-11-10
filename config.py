"""Загрузка и проверка настроек приложения из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Загружаем .env из корня проекта, не перезаписывая уже заданные переменные.
load_dotenv(BASE_DIR / ".env", override=False)


class ConfigError(ValueError):
    """Ошибка в конфигурации приложения."""


@dataclass(frozen=True)
class Settings:
    """Проверенные настройки приложения."""

    vk_group_token: str
    gigachat_credentials: str

    vk_group_id: int | None = None
    vk_transport: str = "auto"
    contact_form_url: str | None = None

    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"
    gigachat_base_url: str = "https://api.giga.chat/v1"
    gigachat_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_verify_ssl: bool = True
    gigachat_ca_bundle: str | None = None
    gigachat_temperature: float = 0.55
    gigachat_max_tokens: int = 800

    max_history_messages: int = 12
    dialog_ttl_minutes: int = 180
    log_level: str = "INFO"


def _read(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _read_required(name: str, description: str) -> str:
    value = _read(name)
    if not value:
        raise ConfigError(
            f"Не задана обязательная переменная {name} ({description}). "
            f"Скопируйте .env.example в .env и заполните значение."
        )
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = _read(name)
    if raw is None:
        return default

    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on", "да"}:
        return True
    if normalized in {"0", "false", "no", "off", "нет"}:
        return False

    raise ConfigError(
        f"Переменная {name} должна быть true или false, получено: {raw!r}."
    )


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw = _read(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть целым числом.") from exc
    if value < minimum:
        raise ConfigError(f"Переменная {name} должна быть не меньше {minimum}.")
    return value


def _read_float(
    name: str, default: float, minimum: float, maximum: float
) -> float:
    raw = _read(name)
    if raw is None:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть числом.") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"Переменная {name} должна быть в диапазоне "
            f"от {minimum} до {maximum}."
        )
    return value


def _read_url(name: str, default: str | None = None) -> str | None:
    value = _read(name, default)
    if value is not None and not value.startswith(("http://", "https://")):
        raise ConfigError(
            f"Переменная {name} должна содержать полный URL, "
            f"начинающийся с http:// или https://."
        )
    return value


def load_settings() -> Settings:
    """Читает переменные окружения и возвращает проверенные настройки."""

    vk_group_token = _read_required(
        "VK_GROUP_TOKEN",
        "ключ доступа сообщества ВКонтакте",
    )
    gigachat_credentials = _read_required(
        "GIGACHAT_CREDENTIALS",
        "Authorization Key из GigaChat Studio",
    )

    # Некоторые пользователи копируют ключ вместе с префиксом Basic.
    if gigachat_credentials.lower().startswith("basic "):
        gigachat_credentials = gigachat_credentials[6:].strip()
    if not gigachat_credentials:
        raise ConfigError("Переменная GIGACHAT_CREDENTIALS пуста.")

    ca_bundle = _read("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        ca_path = Path(ca_bundle).expanduser()
        if not ca_path.is_absolute():
            ca_path = BASE_DIR / ca_path
        if not ca_path.is_file():
            raise ConfigError(
                f"Файл сертификата из GIGACHAT_CA_BUNDLE не найден: {ca_path}"
            )
        ca_bundle = str(ca_path)

    contact_form_url = _read_url("CONTACT_FORM_URL")

    vk_transport = (_read("VK_TRANSPORT", "auto") or "auto").lower()
    if vk_transport not in {"auto", "longpoll", "polling"}:
        raise ConfigError(
            "VK_TRANSPORT должен быть одним из: auto, longpoll, polling."
        )

    vk_group_id: int | None = None
    if _read("VK_GROUP_ID"):
        vk_group_id = _read_int("VK_GROUP_ID", 1, minimum=1)

    max_history_messages = _read_int("MAX_HISTORY_MESSAGES", 12, minimum=2)
    if max_history_messages % 2 != 0:
        # История хранится парами «вопрос — ответ», поэтому размер должен быть чётным.
        max_history_messages -= 1

    settings = Settings(
        vk_group_token=vk_group_token,
        gigachat_credentials=gigachat_credentials,
        vk_group_id=vk_group_id,
        vk_transport=vk_transport,
        contact_form_url=contact_form_url,
        gigachat_scope=_read("GIGACHAT_SCOPE", "GIGACHAT_API_PERS") or "GIGACHAT_API_PERS",
        gigachat_model=_read("GIGACHAT_MODEL", "GigaChat-2") or "GigaChat-2",
        gigachat_base_url=(
            _read_url("GIGACHAT_BASE_URL", "https://api.giga.chat/v1")
            or "https://api.giga.chat/v1"
        ).rstrip("/"),
        gigachat_oauth_url=(
            _read_url(
                "GIGACHAT_OAUTH_URL",
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            )
            or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        ),
        gigachat_verify_ssl=_read_bool("GIGACHAT_VERIFY_SSL", True),
        gigachat_ca_bundle=ca_bundle,
        gigachat_temperature=_read_float(
            "GIGACHAT_TEMPERATURE", 0.55, minimum=0.0, maximum=2.0
        ),
        gigachat_max_tokens=_read_int("GIGACHAT_MAX_TOKENS", 800, minimum=64),
        max_history_messages=max_history_messages,
        dialog_ttl_minutes=_read_int(
            "DIALOG_TTL_MINUTES", 180, minimum=1
        ),
        log_level=(_read("LOG_LEVEL", "INFO") or "INFO").upper(),
    )

    if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(
            "LOG_LEVEL должен быть одним из: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    return settings
