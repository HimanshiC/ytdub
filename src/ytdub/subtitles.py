"""Manual YouTube subtitle selection, download, parsing, and normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
import json
import logging
from pathlib import Path
import re
from typing import Iterable
from xml.etree import ElementTree

from .errors import RuntimeDependencyError, SubtitleParseError, TranscriptUnavailable
from .models import Segment, VideoMetadata


LOGGER = logging.getLogger(__name__)

_TIMESTAMP_LINE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    r"\s+-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    r"(?:\s+.*)?$"
)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class TranscriptCue:
    """A source caption/ASR utterance before conversion to canonical segments."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, float | str]) -> "TranscriptCue":
        return cls(start=float(value["start"]), end=float(value["end"]), text=str(value["text"]))


def choose_manual_subtitle_language(
    video: VideoMetadata, preferred_language: str | None
) -> str | None:
    """Choose a manual subtitle language deterministically, never automatic captions."""

    available = sorted(video.subtitles)
    if not available:
        return None

    preferences = [
        preferred_language,
        video.original_language,
        video.language,
    ]
    for preference in preferences:
        selected = _find_language_match(available, preference)
        if selected:
            return selected
    return available[0]


def _find_language_match(available: list[str], preference: str | None) -> str | None:
    if not preference:
        return None
    normalized = preference.casefold()
    for language in available:
        if language.casefold() == normalized:
            return language
    base = normalized.split("-")[0]
    for language in available:
        if language.casefold().split("-")[0] == base:
            return language
    return None


def parse_subtitle_text(contents: str) -> list[TranscriptCue]:
    """Parse timed WebVTT or SRT cue text and reject unusable caption files."""

    lines = contents.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[TranscriptCue] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            index += 1
            continue

        timestamp_match = _TIMESTAMP_LINE.match(line)
        if "-->" in line and timestamp_match is None:
            raise SubtitleParseError(f"malformed subtitle timestamp line: {line!r}")
        if timestamp_match is None:
            # SRT/VTT cue identifiers precede the timestamp line.
            index += 1
            continue

        start = _parse_timestamp(timestamp_match.group("start"))
        end = _parse_timestamp(timestamp_match.group("end"))
        if end <= start:
            raise SubtitleParseError("subtitle cue end must be after its start")

        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            candidate = lines[index].strip()
            if "-->" in candidate:
                break
            text_lines.append(candidate)
            index += 1
        text = _clean_caption_text(" ".join(text_lines))
        if text:
            cues.append(TranscriptCue(start=start, end=end, text=text))
        if index < len(lines) and "-->" in lines[index]:
            continue
        index += 1

    if not cues:
        raise SubtitleParseError("subtitle file contained no non-empty timed cues")
    return cues


def parse_subtitle_file(path: Path) -> list[TranscriptCue]:
    """Parse one of yt-dlp's common timed manual-subtitle artifact formats."""

    suffix = path.suffix.casefold()
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise SubtitleParseError(f"could not read subtitle file {path.name}: {error}") from error

    if suffix in {".vtt", ".srt"}:
        return parse_subtitle_text(contents)
    if suffix == ".json3":
        return _parse_json3_subtitle_text(contents)
    if suffix in {".ttml", ".xml"}:
        return _parse_ttml_subtitle_text(contents)
    raise SubtitleParseError(f"unsupported subtitle format: {path.suffix}")


def _parse_json3_subtitle_text(contents: str) -> list[TranscriptCue]:
    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as error:
        raise SubtitleParseError(f"malformed JSON3 subtitle data: {error}") from error
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise SubtitleParseError("JSON3 subtitle data did not contain events")

    cues: list[TranscriptCue] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs")
        parts = event.get("segs")
        if not isinstance(start_ms, (int, float)) or not isinstance(duration_ms, (int, float)):
            continue
        if not isinstance(parts, list):
            continue
        text = "".join(
            str(part.get("utf8") or "") for part in parts if isinstance(part, dict)
        )
        if duration_ms > 0 and _clean_caption_text(text):
            cues.append(
                TranscriptCue(
                    start=float(start_ms) / 1000,
                    end=(float(start_ms) + float(duration_ms)) / 1000,
                    text=text,
                )
            )
    if not cues:
        raise SubtitleParseError("JSON3 subtitle data contained no non-empty timed cues")
    return cues


def _parse_ttml_subtitle_text(contents: str) -> list[TranscriptCue]:
    try:
        root = ElementTree.fromstring(contents)
    except ElementTree.ParseError as error:
        raise SubtitleParseError(f"malformed TTML subtitle data: {error}") from error

    cues: list[TranscriptCue] = []
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] != "p":
            continue
        begin = element.get("begin")
        end = element.get("end")
        duration = element.get("dur")
        if begin is None or (end is None and duration is None):
            continue
        start = _parse_ttml_timestamp(begin)
        finish = _parse_ttml_timestamp(end) if end is not None else start + _parse_ttml_timestamp(duration)
        text = "".join(element.itertext())
        if finish > start and _clean_caption_text(text):
            cues.append(TranscriptCue(start=start, end=finish, text=text))
    if not cues:
        raise SubtitleParseError("TTML subtitle data contained no non-empty timed cues")
    return cues


def _parse_ttml_timestamp(value: str | None) -> float:
    if value is None:
        raise SubtitleParseError("missing TTML timestamp")
    normalized = value.strip()
    if normalized.endswith("s"):
        try:
            return float(normalized[:-1])
        except ValueError as error:
            raise SubtitleParseError(f"invalid TTML timestamp: {value!r}") from error
    return _parse_timestamp(normalized)


def _parse_timestamp(value: str) -> float:
    components = value.replace(",", ".").split(":")
    try:
        if len(components) == 3:
            hours, minutes, seconds = components
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(components) == 2:
            minutes, seconds = components
            return int(minutes) * 60 + float(seconds)
    except ValueError as error:
        raise SubtitleParseError(f"invalid subtitle timestamp: {value!r}") from error
    raise SubtitleParseError(f"invalid subtitle timestamp: {value!r}")


def _clean_caption_text(value: str) -> str:
    return _WHITESPACE.sub(" ", unescape(_TAG.sub("", value))).strip()


def normalize_cues(cues: Iterable[TranscriptCue]) -> list[Segment]:
    """Create the canonical timestamped Segment representation from source cues."""

    cleaned: list[TranscriptCue] = []
    seen: set[tuple[int, int, str]] = set()
    for cue in cues:
        text = _clean_caption_text(cue.text)
        if cue.start < 0 or cue.end <= cue.start or not text:
            continue
        dedupe_key = (round(cue.start * 1000), round(cue.end * 1000), text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(TranscriptCue(start=cue.start, end=cue.end, text=text))

    cleaned.sort(key=lambda cue: (cue.start, cue.end, cue.text))
    if not cleaned:
        raise SubtitleParseError("no usable timestamped transcript cues remain after normalization")

    return [
        Segment(
            segment_id=f"seg-{position:06d}-{round(cue.start * 1000):010d}",
            start=cue.start,
            end=cue.end,
            source_text=cue.text,
        )
        for position, cue in enumerate(cleaned, start=1)
    ]


class ManualSubtitleProvider:
    """Fetch a selected manually uploaded subtitle track through yt-dlp."""

    name = "youtube_manual_subtitles"

    def fetch(
        self,
        video: VideoMetadata,
        artifact_directory: Path,
        preferred_language: str | None,
    ) -> tuple[str, list[TranscriptCue], Path]:
        """Return selected language, parsed cues, and persisted raw subtitle file."""

        language = choose_manual_subtitle_language(video, preferred_language)
        if language is None:
            raise TranscriptUnavailable("no manually uploaded subtitle tracks are available")

        yt_dlp = _import_yt_dlp()
        artifact_directory.mkdir(parents=True, exist_ok=True)
        before_download = set(artifact_directory.iterdir())
        LOGGER.info("Captions: downloading manual %s subtitle track", language)
        options = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": False,
            "subtitleslangs": [language],
            "subtitlesformat": "vtt/srt/json3/ttml/best",
            "outtmpl": {"default": str(artifact_directory / "manual.%(ext)s")},
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.extract_info(video.webpage_url, download=True)
        except Exception as error:  # yt-dlp exposes downloader-specific error classes.
            raise TranscriptUnavailable(
                f"manual subtitle download failed for language {language!r}: {error}"
            ) from error

        created_files = sorted(
            path
            for path in artifact_directory.iterdir()
            if path not in before_download
            and path.is_file()
            and path.suffix.lower() in {".vtt", ".srt", ".json3", ".ttml", ".xml"}
        )
        if not created_files:
            raise TranscriptUnavailable(
                f"selected manual subtitle track {language!r} was missing after download"
            )

        subtitle_path = created_files[0]
        cues = parse_subtitle_file(subtitle_path)
        return language, cues, subtitle_path


def _import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeDependencyError(
            "yt-dlp is required for YouTube ingestion. Install requirements.txt first."
        ) from error
    return yt_dlp
