"""Translation backend abstraction for baseline and improved dubbing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Protocol

from groq import Groq

from .models import Segment


LOGGER = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
BASELINE_MAX_CONCURRENCY = 3
MAX_RETRIES = 2


class TranslationBackend(Protocol):
    """Common interface implemented by translation backends."""

    def translate_segment(
        self,
        segment: Segment,
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> str:
        ...

    def translate_segments(
        self,
        segments: list[Segment],
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        ...


@dataclass(frozen=True, slots=True)
class GroqTranslationConfig:
    """Configuration for the Groq translation backend."""

    model: str = DEFAULT_GROQ_MODEL
    temperature: float = 0.2
    max_retries: int = MAX_RETRIES


class GroqTranslator:
    """
    Groq-backed multilingual translator.

    Baseline mode translates each canonical segment independently.
    Improved mode can provide contextual information while keeping the same
    segment-level interface.
    """

    def __init__(
        self,
        config: GroqTranslationConfig | None = None,
    ) -> None:
        self.config = config or GroqTranslationConfig()

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it in the environment before "
                "running the dubbing pipeline."
            )

        self.client = Groq()

    def translate_segment(
        self,
        segment: Segment,
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """Translate one segment with bounded retry/backoff."""

        system_prompt = _build_system_prompt(
            context=context,
            glossary=glossary,
        )

        user_prompt = (
            "Translate the following source speech into natural spoken English.\n\n"
            f"Source segment:\n{segment.source_text}\n\n"
            "Return ONLY the English translation. "
            "Do not add explanations, labels, quotation marks, or commentary."
        )

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    reasoning_effort="low",
                    include_reasoning=False,
                    max_completion_tokens=512,
                )

                content = response.choices[0].message.content

                if not content or not content.strip():
                    raise RuntimeError(
                        "Groq returned an empty translation response"
                    )

                translated = content.strip()

                if translated.startswith("```"):
                    translated = _strip_code_fence(translated)

                if not translated:
                    raise RuntimeError(
                        "Groq returned an empty translation response"
                    )

                return translated

            except Exception as error:
                last_error = error

                if attempt >= self.config.max_retries:
                    break

                delay = 2**attempt
                LOGGER.warning(
                    "Translation retry segment=%s attempt=%d/%d "
                    "in=%ds: %s",
                    segment.segment_id,
                    attempt + 1,
                    self.config.max_retries + 1,
                    delay,
                    error,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"translation failed for {segment.segment_id}: {last_error}"
        ) from last_error

    def translate_segments(
        self,
        segments: list[Segment],
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Translate segments independently.

        The baseline intentionally uses one request per segment so every
        translation has an exact one-to-one relationship with its canonical
        segment. The improved mode can later override this behavior with
        contextual chunk translation without changing the caller interface.
        """

        if not segments:
            return []

        # Keep the API contract simple and deterministic for the baseline.
        # DubbingPipeline currently calls this sequentially; concurrency can
        # be introduced later after the core path is validated.
        results: list[str] = []

        for index, segment in enumerate(segments, start=1):
            LOGGER.info(
                "Translation request [%d/%d] segment=%s",
                index,
                len(segments),
                segment.segment_id,
            )

            results.append(
                self.translate_segment(
                    segment,
                    context=context,
                    glossary=glossary,
                )
            )

        return results


def _build_system_prompt(
    *,
    context: str | None,
    glossary: dict[str, str] | None,
) -> str:
    prompt = (
        "You are a professional audiovisual translator. "
        "Translate speech into natural, concise spoken English while preserving "
        "the original meaning, speaker intent, names, numbers, and factual details. "
        "Do not invent information."
    )

    if context:
        prompt += (
            "\n\nUse the following surrounding dialogue only as context for "
            "resolving references, terminology, and meaning. "
            "Translate only the requested source segment.\n"
            f"{context}"
        )

    if glossary:
        prompt += "\n\nUse these terminology mappings consistently:\n"
        for source, target in glossary.items():
            prompt += f"- {source} -> {target}\n"

    return prompt


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def parse_translation_response(content: str) -> str:
    """
    Normalize a plain-text translation response.

    Kept separate so the improved contextual backend can reuse response
    validation later without changing the public translator interface.
    """

    text = content.strip()

    if not text:
        raise RuntimeError("translation response is empty")

    return text