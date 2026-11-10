"""Загрузка базы знаний и подбор релевантных разделов под вопрос пользователя."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
WORD_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)

# Небольшой список служебных слов, которые не помогают выбрать раздел.
STOP_WORDS = {
    "этот",
    "эта",
    "это",
    "эти",
    "какой",
    "какая",
    "какие",
    "каких",
    "который",
    "которая",
    "которые",
    "можно",
    "нужно",
    "надо",
    "очень",
    "просто",
    "расскажи",
    "расскажите",
    "подскажи",
    "подскажите",
    "пожалуйста",
    "почему",
    "зачем",
    "когда",
    "сколько",
    "где",
    "есть",
    "быть",
    "был",
    "была",
    "были",
    "для",
    "или",
    "как",
    "что",
    "чем",
    "при",
    "над",
    "под",
    "без",
    "про",
    "все",
    "весь",
    "всё",
    "она",
    "они",
    "оно",
    "его",
    "её",
    "их",
    "мне",
    "меня",
    "вас",
    "вам",
    "ваш",
    "ваша",
    # Слова про енотов есть почти в каждом разделе, поэтому они не помогают
    # выбрать нужный раздел.
    "енот",
    "енота",
    "еноту",
    "енотом",
    "еноте",
    "еноты",
    "енотов",
    "енотам",
    "енотами",
    "енотах",
}

ALWAYS_INCLUDE = {"кратко о виде", "границы базы знаний"}


@dataclass(frozen=True)
class KnowledgeSection:
    """Один раздел базы знаний."""

    title: str
    body: str

    @property
    def text(self) -> str:
        return f"## {self.title}\n\n{self.body}".strip()


class KnowledgeBase:
    """База знаний в формате Markdown с простым поиском по разделам."""

    def __init__(self, path: str | Path, max_context_chars: int = 20_000) -> None:
        self.path = Path(path)
        self.max_context_chars = max_context_chars
        self.raw_text = self._load()
        self.intro, self.sections = self._parse(self.raw_text)

        if not self.raw_text.strip():
            raise ValueError(f"База знаний пуста: {self.path}")
        if not self.sections:
            raise ValueError(
                f"В базе знаний нет разделов, начинающихся с '##': {self.path}"
            )

    def _load(self) -> str:
        if not self.path.is_file():
            raise FileNotFoundError(f"Файл базы знаний не найден: {self.path}")
        return self.path.read_text(encoding="utf-8-sig").strip()

    @staticmethod
    def _parse(text: str) -> tuple[str, list[KnowledgeSection]]:
        matches = list(SECTION_RE.finditer(text))
        intro = text[: matches[0].start()].strip() if matches else text.strip()

        sections: list[KnowledgeSection] = []
        for index, match in enumerate(matches):
            body_start = match.end()
            body_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )
            body = text[body_start:body_end].strip()
            sections.append(
                KnowledgeSection(title=match.group("title").strip(), body=body)
            )

        return intro, sections

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.lower().replace("ё", "е")
        return [
            word
            for word in WORD_RE.findall(normalized)
            if word not in STOP_WORDS
        ]

    @classmethod
    def _score(cls, query_tokens: set[str], section: KnowledgeSection) -> int:
        if not query_tokens:
            return 0

        title_words = set(cls._tokens(section.title))
        body_words = set(cls._tokens(section.body))
        body_normalized = section.body.lower().replace("ё", "е")

        score = 0
        for token in query_tokens:
            # Грубое усечение помогает сопоставлять словоформы:
            # «питается» → «пита» → «питание», «питании».
            prefix = token[:4] if len(token) >= 5 else token
            if token in title_words or any(
                word.startswith(prefix) for word in title_words
            ):
                score += 8
            if any(word.startswith(prefix) for word in body_words):
                score += 2
            # Ограничиваем вклад одного слова, чтобы большие разделы
            # не выигрывали только за счёт размера.
            score += min(body_normalized.count(token), 4)

        return score

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text

        chunk = text[:limit]
        cut = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind(" "))
        if cut > limit // 2:
            chunk = chunk[:cut].rstrip()
        return chunk.rstrip() + "…"

    def build_context(self, query: str, max_chars: int | None = None) -> str:
        """Возвращает базу знаний или её релевантную часть для промпта.

        Пока база целиком помещается в лимит, она передаётся модели полностью.
        Если база вырастет, будут выбраны наиболее подходящие разделы.
        """

        limit = max_chars or self.max_context_chars
        if len(self.raw_text) <= limit:
            return self.raw_text

        query_tokens = set(self._tokens(query))
        always = [
            section
            for section in self.sections
            if section.title.lower() in ALWAYS_INCLUDE
        ]
        candidates = [
            section
            for section in self.sections
            if section.title.lower() not in ALWAYS_INCLUDE
        ]
        candidates.sort(key=lambda section: self._score(query_tokens, section), reverse=True)

        selected: list[str] = []
        seen: set[str] = set()
        used = 0

        def add(section: KnowledgeSection) -> bool:
            nonlocal used
            if section.title in seen:
                return True

            separator = 2 if selected else 0
            available = limit - used - separator
            if available <= 0:
                return False

            text = section.text
            if len(text) > available:
                text = self._truncate(text, available)
            if not text:
                return False

            selected.append(text)
            seen.add(section.title)
            used += len(text) + separator
            return True

        for section in always:
            add(section)

        for section in candidates:
            if used >= limit:
                break
            add(section)

        parts: list[str] = []
        if self.intro:
            parts.append(self.intro)
        parts.extend(selected)
        return "\n\n".join(parts).strip()
