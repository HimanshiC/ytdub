"""Transcript-provider abstraction and faster-whisper fallback provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
import re
from typing import Protocol

from .errors import RuntimeDependencyError, TranscriptUnavailable
from .models import VideoMetadata
from .subtitles import ManualSubtitleProvider, TranscriptCue


LOGGER = logging.getLogger(__name__)

# These boundaries are deliberately source-side, deterministic, and independent
# of future translation or TTS timing logic.
WHISPER_WORD_TIMESTAMP_STRATEGY = "word-timestamps-v2"
SILENCE_BOUNDARY_SECONDS = 0.75
MAX_UTTERANCE_SECONDS = 12.0
_TERMINAL_PUNCTUATION = (".", "!", "?", "…", "。", "！", "？")
_WHITESPACE = re.compile(r"\s+")
_INITIAL_ABBREVIATION = re.compile(r"^(?:[A-Za-zÀ-ÖØ-öø-ÿ]\.)+$")
_NON_TERMINAL_ABBREVIATIONS = frozenset(
    {
        "bzw.",
        "ca.",
        "d.h.",
        "dr.",
        "e.g.",
        "etc.",
        "mr.",
        "mrs.",
        "nr.",
        "prof.",
        "u.a.",
        "usw.",
        "z.b.",
    }
)


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """Provider transcript persisted before canonical Segment normalization.

    ``raw_cues`` retains coarse provider units for diagnostics. ``cues`` carries
    the timestamp-refined utterances that will become canonical segments.
    """

    provider: str
    language: str | None
    language_probability: float | None
    cues: tuple[TranscriptCue, ...]
    raw_artifact_path: str | None = None
    raw_cues: tuple[TranscriptCue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "language": self.language,
            "language_probability": self.language_probability,
            "cues": [cue.to_dict() for cue in self.cues],
            "raw_artifact_path": self.raw_artifact_path,
            "raw_cues": [cue.to_dict() for cue in self.raw_cues],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TranscriptResult":
        raw_cues = value.get("cues")
        if not isinstance(raw_cues, list):
            raise ValueError("cached transcript did not contain a cue list")
        raw_provider_cues = value.get("raw_cues", raw_cues)
        if not isinstance(raw_provider_cues, list):
            raise ValueError("cached transcript raw_cues must be a cue list")
        return cls(
            provider=str(value["provider"]),
            language=str(value["language"]) if value.get("language") else None,
            language_probability=(
                float(value["language_probability"])
                if value.get("language_probability") is not None
                else None
            ),
            cues=tuple(TranscriptCue.from_dict(cue) for cue in raw_cues),  # type: ignore[arg-type]
            raw_artifact_path=(
                str(value["raw_artifact_path"]) if value.get("raw_artifact_path") else None
            ),
            raw_cues=tuple(
                TranscriptCue.from_dict(cue) for cue in raw_provider_cues  # type: ignore[arg-type]
            ),
        )


class TranscriptProvider(Protocol):
    """A source transcript backend that returns timestamped cues."""

    name: str

    def fetch(
        self,
        video: VideoMetadata,
        audio_path: Path,
        artifact_directory: Path,
        preferred_language: str | None,
    ) -> TranscriptResult:
        """Produce a usable timestamped transcript or raise TranscriptUnavailable."""


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    """Explicit faster-whisper configuration included in transcript cache keys."""

    model: str = "small"
    device: str = "auto"
    compute_type: str = "int8"

    def to_dict(self) -> dict[str, object]:
        """Include segmentation strategy in cache invalidation inputs."""

        return {
            **asdict(self),
            "word_timestamps": True,
            "vad_filter": False,
            "utterance_refinement": {
                "strategy": WHISPER_WORD_TIMESTAMP_STRATEGY,
                "silence_boundary_seconds": SILENCE_BOUNDARY_SECONDS,
                "max_utterance_seconds": MAX_UTTERANCE_SECONDS,
            },
        }


@dataclass(frozen=True, slots=True)
class TimedWord:
    """A word-level ASR token whose times are measured from source audio."""

    start: float
    end: float
    text: str


def refine_timed_words(words: tuple[TimedWord, ...]) -> list[TranscriptCue]:
    """Split word-timestamped ASR output into compact natural utterances.

    Every output boundary is an observed word boundary. Sentence punctuation and
    source silence are preferred. The maximum window only prevents an unusually
    long, punctuation-free ASR decoder segment from reaching downstream TTS.
    """

    usable_words = tuple(word for word in words if _is_usable_word(word))
    if not usable_words:
        return []

    cues: list[TranscriptCue] = []
    current: list[TimedWord] = []
    for word in usable_words:
        if current:
            previous = current[-1]
            silence_gap = word.start - previous.end
            exceeds_maximum = word.end - current[0].start > MAX_UTTERANCE_SECONDS
            if silence_gap >= SILENCE_BOUNDARY_SECONDS or exceeds_maximum:
                cues.append(_cue_from_words(current))
                current = []

        current.append(word)
        if _ends_utterance(word.text):
            cues.append(_cue_from_words(current))
            current = []

    if current:
        cues.append(_cue_from_words(current))
    return cues


def _is_usable_word(word: TimedWord) -> bool:
    return (
        bool(word.text.strip())
        and math.isfinite(word.start)
        and math.isfinite(word.end)
        and word.start >= 0
        and word.end > word.start
    )


def _cue_from_words(words: list[TimedWord]) -> TranscriptCue:
    text = _WHITESPACE.sub(" ", "".join(word.text for word in words)).strip()
    if not text:
        text = " ".join(word.text.strip() for word in words)
    return TranscriptCue(start=words[0].start, end=words[-1].end, text=text)


def _ends_utterance(word_text: str) -> bool:
    """Recognize terminal punctuation without splitting common abbreviations."""

    token = word_text.strip()
    if not token.endswith(_TERMINAL_PUNCTUATION):
        return False
    if not token.endswith("."):
        return True
    normalized = token.casefold()
    return not (
        _INITIAL_ABBREVIATION.fullmatch(token)
        or normalized in _NON_TERMINAL_ABBREVIATIONS
    )


def _timed_words_from_whisper(segment: object) -> tuple[TimedWord, ...]:
    """Extract valid word timestamps without depending on Whisper's runtime types."""

    raw_words = getattr(segment, "words", None) or ()
    timed_words: list[TimedWord] = []
    for raw_word in raw_words:
        try:
            timed_words.append(
                TimedWord(
                    start=float(raw_word.start),
                    end=float(raw_word.end),
                    text=str(raw_word.word),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(timed_words)


class FasterWhisperProvider:
    """ASR fallback used only when manual subtitles are unavailable or invalid."""

    name = "faster_whisper"

    def __init__(self, settings: WhisperSettings) -> None:
        self.settings = settings

    def fetch(
        self,
        video: VideoMetadata,
        audio_path: Path,
        artifact_directory: Path,
        preferred_language: str | None,
    ) -> TranscriptResult:
        del video, artifact_directory
        if not audio_path.is_file():
            raise TranscriptUnavailable(f"audio artifact is unavailable: {audio_path}")
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeDependencyError(
                "faster-whisper is required when manual subtitles cannot be used. "
                "Install requirements.txt first."
            ) from error

        LOGGER.info(
            "Transcript: loading faster-whisper model=%s device=%s compute_type=%s",
            self.settings.model,
            self.settings.device,
            self.settings.compute_type,
        )
        try:
            model = WhisperModel(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
            generated_segments, info = model.transcribe(
                str(audio_path),
                language=preferred_language or None,
                beam_size=5,
                word_timestamps=True,
                # The lightweight default VAD changed recognition content during
                # the Milestone 1 smoke test. Word timestamps supply observed
                # silence boundaries without filtering source speech away.
                vad_filter=False,
            )
            raw_cues: list[TranscriptCue] = []
            cues: list[TranscriptCue] = []
            timestamped_word_count = 0
            for segment in generated_segments:
                raw_cue = TranscriptCue(
                    start=float(segment.start), end=float(segment.end), text=segment.text
                )
                raw_cues.append(raw_cue)
                words = _timed_words_from_whisper(segment)
                timestamped_word_count += len(words)
                refined_cues = refine_timed_words(words)
                cues.extend(refined_cues or [raw_cue])
        except Exception as error:  # Model/download/ffmpeg backend errors are surfaced clearly.
            raise TranscriptUnavailable(f"faster-whisper transcription failed: {error}") from error
        if not cues:
            raise TranscriptUnavailable("faster-whisper returned no timestamped speech segments")
        language = getattr(info, "language", None) or preferred_language
        language_probability = getattr(info, "language_probability", None)
        LOGGER.info(
            "Transcript: faster-whisper raw_segments=%d timestamped_words=%d "
            "refined_utterances=%d strategy=%s vad=disabled",
            len(raw_cues),
            timestamped_word_count,
            len(cues),
            WHISPER_WORD_TIMESTAMP_STRATEGY,
        )
        return TranscriptResult(
            provider=self.name,
            language=language,
            language_probability=(
                float(language_probability) if language_probability is not None else None
            ),
            cues=tuple(cues),
            raw_cues=tuple(raw_cues),
        )


class PreferredTranscriptProvider:
    """Use manual subtitles first and deterministically fall back to ASR."""

    def __init__(self, whisper: FasterWhisperProvider) -> None:
        self.manual = ManualSubtitleProvider()
        self.whisper = whisper

    def fetch(
        self,
        video: VideoMetadata,
        audio_path: Path,
        artifact_directory: Path,
        preferred_language: str | None,
    ) -> TranscriptResult:
        try:
            language, cues, subtitle_path = self.manual.fetch(
                video=video,
                artifact_directory=artifact_directory / "raw_subtitles",
                preferred_language=preferred_language,
            )
            LOGGER.info("Transcript: accepted %d manual subtitle cues (%s)", len(cues), language)
            return TranscriptResult(
                provider=self.manual.name,
                language=language,
                language_probability=None,
                cues=tuple(cues),
                raw_artifact_path=str(subtitle_path),
                raw_cues=tuple(cues),
            )
        except TranscriptUnavailable as error:
            LOGGER.warning("Transcript: manual subtitles unusable (%s); falling back to faster-whisper", error)
            return self.whisper.fetch(video, audio_path, artifact_directory, preferred_language)
