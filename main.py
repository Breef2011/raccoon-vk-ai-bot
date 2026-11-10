"""Точка входа чат-бота сообщества «Еноты»."""

from __future__ import annotations

import argparse
import logging
import sys

from bot.dialog_manager import DialogService, DialogStore
from bot.gigachat_client import GigaChatClient, GigaChatError
from bot.knowledge_base import KnowledgeBase
from bot.vk_bot import VkBot
from config import BASE_DIR, ConfigError, Settings, load_settings


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build_gigachat_client(settings: Settings) -> GigaChatClient:
    return GigaChatClient(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
        model=settings.gigachat_model,
        base_url=settings.gigachat_base_url,
        oauth_url=settings.gigachat_oauth_url,
        verify_ssl=settings.gigachat_verify_ssl,
        ca_bundle=settings.gigachat_ca_bundle,
        temperature=settings.gigachat_temperature,
        max_tokens=settings.gigachat_max_tokens,
    )


def run_checks(settings: Settings) -> int:
    """Проверяет доступы к GigaChat и ВКонтакте без запуска бота."""

    client = build_gigachat_client(settings)
    ok = True

    for label, path in (
        ("база знаний", BASE_DIR / "knowledge" / "raccoon_knowledge.md"),
        ("системный промпт", BASE_DIR / "prompts" / "system_prompt.txt"),
    ):
        if not path.is_file():
            ok = False
            print(f"  {label}: ОШИБКА — файл не найден: {path}", file=sys.stderr)

    prompt_path = BASE_DIR / "prompts" / "system_prompt.txt"
    if prompt_path.is_file():
        prompt_text = prompt_path.read_text(encoding="utf-8-sig")
        if "{{KNOWLEDGE_BASE}}" not in prompt_text:
            ok = False
            print(
                "  системный промпт: ОШИБКА — нет плейсхолдера "
                "{{KNOWLEDGE_BASE}} для базы знаний.",
                file=sys.stderr,
            )

    print("Проверяю GigaChat...", flush=True)
    try:
        models = client.check_connection()
        if models:
            shown = ", ".join(models[:10])
            print(f"  GigaChat: OK. Доступные модели: {shown}", flush=True)
        else:
            print("  GigaChat: OK, токен получен.", flush=True)
    except GigaChatError as exc:
        ok = False
        print(f"  GigaChat: ОШИБКА — {exc}", file=sys.stderr, flush=True)
    finally:
        client.close()

    print("Проверяю ВКонтакте...", flush=True)
    try:
        group_id, group_name = VkBot.check_connection(
            settings.vk_group_token,
            settings.vk_group_id,
        )
        print(f"  ВКонтакте: OK. Сообщество «{group_name}», id={group_id}.", flush=True)
    except Exception as exc:
        ok = False
        print(f"  ВКонтакте: ОШИБКА — {exc}", file=sys.stderr, flush=True)

    if ok:
        print("\nВсе проверки пройдены. Можно запускать бота без флага --check.")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Чат-бот сообщества «Еноты» с GigaChat API.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="проверить настройки и доступы, не запуская обработку сообщений",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    setup_logging(settings.log_level)

    if not settings.gigachat_verify_ssl:
        logging.getLogger(__name__).warning(
            "Проверка TLS-сертификатов GigaChat отключена. "
            "Используйте это только для локальной диагностики."
        )

    if args.check:
        return run_checks(settings)

    try:
        knowledge_base = KnowledgeBase(
            BASE_DIR / "knowledge" / "raccoon_knowledge.md"
        )
        store = DialogStore(
            max_messages=settings.max_history_messages,
            ttl_minutes=settings.dialog_ttl_minutes,
        )
        client = build_gigachat_client(settings)
        try:
            dialog_service = DialogService(
                client=client,
                knowledge_base=knowledge_base,
                system_prompt_path=BASE_DIR / "prompts" / "system_prompt.txt",
                contact_form_url=settings.contact_form_url,
                store=store,
            )
        except (FileNotFoundError, ValueError):
            client.close()
            raise
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка подготовки бота: {exc}", file=sys.stderr)
        return 1

    try:
        bot = VkBot(
            token=settings.vk_group_token,
            group_id=settings.vk_group_id,
            dialog_service=dialog_service,
            transport=settings.vk_transport,
        )
        bot.run()
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except Exception:
        logging.getLogger(__name__).exception("Непредвиденная ошибка при запуске бота.")
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
