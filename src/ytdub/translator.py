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
import subprocess
from pathlib import Path

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
            "Translate the entire source passage into natural spoken English.\n\n"

            "The passage may contain transcription artifacts or awkward "
            "sentence segmentation. Reconstruct the intended meaning from "
            "the full passage rather than translating isolated fragments.\n\n"

            "Use natural conversational English suitable for voice dubbing. "
            "Preserve all substantive information. Do not add information "
            "that is not supported by the source.\n\n"

            f"Source passage:\n{segment.source_text}\n\n"

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

# ---------------------------------------------------------------------------
# IndicTrans2 backend
# ---------------------------------------------------------------------------

INDIC_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

INDIC_LANGUAGE_CODES = {
    "hi": "hin_Deva",
}


class IndicTrans2Translator:
    """
    IndicTrans2 backend isolated in .venv-indic.

    The main application remains in the normal .venv and communicates with
    the IndicTrans2 environment through one subprocess per translation batch.
    """

    def __init__(
        self,
        *,
        python_executable: Path | None = None,
        max_chars_per_batch: int = 12000,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

        self.python_executable = python_executable or (
            project_root / ".venv-indic" / "Scripts" / "python.exe"
        )

        self.worker_path = Path(__file__).resolve().with_name(
            "indictrans_worker.py"
        )

        self.max_chars_per_batch = max_chars_per_batch

        if not self.python_executable.exists():
            raise RuntimeError(
                f"IndicTrans2 Python environment not found: "
                f"{self.python_executable}"
            )

        if not self.worker_path.exists():
            raise RuntimeError(
                f"IndicTrans2 worker not found: {self.worker_path}"
            )

    def translate_segment(
        self,
        segment: Segment,
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
        source_language: str = "hi",
    ) -> str:
        translations = self.translate_segments(
            [segment],
            context=context,
            glossary=glossary,
            source_language=source_language,
        )

        return translations[0]
        def rewrite_for_duration(
            self,
            segment: Segment,
            current_translation: str,
            *,
            target_duration: float,
            actual_duration: float,
            context: str | None = None,
            glossary: dict[str, str] | None = None,
        ) -> str:
            """
            Rewrite a translation when its synthesized speech is substantially
            shorter than the available source speech duration.

            The model is asked to preserve meaning while producing a fuller,
            natural spoken rendering. Python remains responsible for measuring
            the resulting audio duration.
            """

            system_prompt = _build_system_prompt(
                context=context,
                glossary=glossary,
            )

            user_prompt = (
                "Revise the English dubbing passage so it is naturally fuller "
                "and better suited to the available speaking time.\n\n"

                f"Target speaking duration: approximately {target_duration:.1f} seconds.\n"
                f"Current synthesized duration: approximately {actual_duration:.1f} seconds.\n\n"

                "Preserve the complete meaning of the original source passage. "
                "Do not invent facts, names, events, numbers, or explanations. "
                "Do not repeat ideas merely to make the passage longer. "
                "Use natural conversational English suitable for spoken dubbing. "
                "Preserve uncertainty where the source is uncertain. "
                "The result should sound like something a person would naturally say, "
                "not like an expanded written translation.\n\n"

                f"Original source passage:\n{segment.source_text}\n\n"

                f"Current English translation:\n{current_translation}\n\n"

                "Return ONLY the revised English passage. "
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
                        max_completion_tokens=640,
                    )

                    content = response.choices[0].message.content

                    if not content or not content.strip():
                        raise RuntimeError(
                            "Groq returned an empty duration-repair response"
                        )

                    rewritten = content.strip()

                    if rewritten.startswith("```"):
                        rewritten = _strip_code_fence(rewritten)

                    if not rewritten:
                        raise RuntimeError(
                            "Groq returned an empty duration-repair response"
                        )

                    return rewritten

                except Exception as error:
                    last_error = error

                    if attempt >= self.config.max_retries:
                        break

                    delay = 2**attempt
                    LOGGER.warning(
                        "Duration repair retry segment=%s attempt=%d/%d "
                        "in=%ds: %s",
                        segment.segment_id,
                        attempt + 1,
                        self.config.max_retries + 1,
                        delay,
                        error,
                    )
                    time.sleep(delay)

            raise RuntimeError(
                f"duration repair failed for {segment.segment_id}: {last_error}"
            ) from last_error
    def translate_segments(
        self,
        segments: list[Segment],
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
        source_language: str = "hi",
    ) -> list[str]:
        if not segments:
            return []

        language = source_language.lower().split("-")[0]

        if language not in INDIC_LANGUAGE_CODES:
            raise RuntimeError(
                f"IndicTrans2 does not have a configured language mapping "
                f"for {source_language}"
            )

        texts = [segment.source_text.strip() for segment in segments]

        if any(not text for text in texts):
            raise RuntimeError(
                "IndicTrans2 cannot translate an empty source segment"
            )

        # Context/glossary are deliberately ignored here in baseline mode.
        # They belong to the improved contextual translator.
        if context or glossary:
            LOGGER.debug(
                "IndicTrans2 baseline backend ignoring context/glossary"
            )

        payload = {
            "source_language": language,
            "texts": texts,
        }

        LOGGER.info(
            "IndicTrans2: translating %d %s segments",
            len(texts),
            language,
        )

        completed = subprocess.run(
            [
                str(self.python_executable),
                str(self.worker_path),
            ],
            input=json.dumps(payload, ensure_ascii=True),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=1800,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip()

            raise RuntimeError(
                "IndicTrans2 worker failed"
                + (f": {stderr}" if stderr else "")
            )

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "IndicTrans2 worker returned invalid JSON"
            ) from error

        translations = result.get("translations")

        if not isinstance(translations, list):
            raise RuntimeError(
                "IndicTrans2 worker response missing 'translations'"
            )

        if len(translations) != len(segments):
            raise RuntimeError(
                "IndicTrans2 returned an unexpected number of translations: "
                f"expected {len(segments)}, got {len(translations)}"
            )

        cleaned: list[str] = []

        for segment, translation in zip(
            segments,
            translations,
            strict=True,
        ):
            if not isinstance(translation, str) or not translation.strip():
                raise RuntimeError(
                    f"IndicTrans2 returned empty translation for "
                    f"{segment.segment_id}"
                )

            cleaned.append(translation.strip())

        return cleaned


class HybridTranslator:
    """
    Route Indic languages to IndicTrans2 and everything else to Groq.

    The source language is supplied by the pipeline, so language routing is
    explicit rather than inferred from the text.
    """

    def __init__(
        self,
        source_language: str,
        *,
        groq_config: GroqTranslationConfig | None = None,
    ) -> None:
        self.source_language = source_language.lower().split("-")[0]

        self.indic = IndicTrans2Translator()
        self.groq = GroqTranslator(groq_config)

    @property
    def provider_name(self) -> str:
        if self.source_language in INDIC_LANGUAGE_CODES:
            return INDIC_MODEL_NAME

        return self.groq.config.model

    def translate_segment(
        self,
        segment: Segment,
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> str:
        if self.source_language in INDIC_LANGUAGE_CODES:
            return self.indic.translate_segment(
                segment,
                context=context,
                glossary=glossary,
                source_language=self.source_language,
            )

        return self.groq.translate_segment(
            segment,
            context=context,
            glossary=glossary,
        )

    def translate_segments(
        self,
        segments: list[Segment],
        *,
        context: str | None = None,
        glossary: dict[str, str] | None = None,
    ) -> list[str]:
        if self.source_language in INDIC_LANGUAGE_CODES:
            return self.indic.translate_segments(
                segments,
                context=context,
                glossary=glossary,
                source_language=self.source_language,
            )

        return self.groq.translate_segments(
            segments,
            context=context,
            glossary=glossary,
        )