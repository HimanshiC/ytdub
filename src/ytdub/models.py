"""Canonical data models shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Segment:
    """A normalized utterance with source-owned, immutable timing metadata.

    Later milestones populate the optional translated/TTS/synchronization fields.
    The ingestion layer only sets the stable ID, source timings, source text, and
    target duration.
    """

    segment_id: str
    start: float
    end: float
    source_text: str
    translated_text: str | None = None
    tts_artifact_path: str | None = None
    target_duration: float | None = None
    actual_tts_duration: float | None = None
    synchronization_action: str | None = None
    synchronization_status: str = "PENDING"
    prosody: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("segment_id must not be empty")
        if self.start < 0:
            raise ValueError("segment start must be non-negative")
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        if not self.source_text.strip():
            raise ValueError("source_text must not be empty")

        expected_duration = self.end - self.start
        if self.target_duration is None:
            self.target_duration = expected_duration
        elif abs(self.target_duration - expected_duration) > 0.001:
            raise ValueError(
                "target_duration must equal end - start; source timing is owned "
                "by deterministic code"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        """Restore a segment persisted by :meth:`to_dict`."""

        return cls(**value)


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    """A manually uploaded subtitle track advertised by YouTube metadata."""

    language: str
    extension: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str | None]) -> "SubtitleTrack":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """The durable subset of yt-dlp metadata required by this pipeline."""

    video_id: str
    webpage_url: str
    title: str
    duration: float | None
    uploader: str | None
    upload_date: str | None
    language: str | None
    original_language: str | None
    subtitles: dict[str, tuple[SubtitleTrack, ...]] = field(default_factory=dict)

    @classmethod
    def from_yt_dlp(cls, info: dict[str, Any], requested_url: str) -> "VideoMetadata":
        video_id = str(info.get("id") or "").strip()
        if not video_id:
            raise ValueError("yt-dlp metadata did not contain a video ID")

        manual_subtitles: dict[str, tuple[SubtitleTrack, ...]] = {}
        subtitle_info = info.get("subtitles") or {}
        if isinstance(subtitle_info, dict):
            for language, formats in subtitle_info.items():
                if not isinstance(language, str) or not isinstance(formats, list):
                    continue
                tracks = tuple(
                    SubtitleTrack(
                        language=language,
                        extension=item.get("ext") if isinstance(item, dict) else None,
                        name=item.get("name") if isinstance(item, dict) else None,
                    )
                    for item in formats
                )
                if tracks:
                    manual_subtitles[language] = tracks

        raw_duration = info.get("duration")
        duration = float(raw_duration) if isinstance(raw_duration, (int, float)) else None
        return cls(
            video_id=video_id,
            webpage_url=str(info.get("webpage_url") or info.get("original_url") or requested_url),
            title=str(info.get("title") or video_id),
            duration=duration,
            uploader=_optional_string(info.get("uploader") or info.get("channel")),
            upload_date=_optional_string(info.get("upload_date")),
            language=_optional_string(info.get("language")),
            original_language=_optional_string(info.get("original_language")),
            subtitles=manual_subtitles,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "webpage_url": self.webpage_url,
            "title": self.title,
            "duration": self.duration,
            "uploader": self.uploader,
            "upload_date": self.upload_date,
            "language": self.language,
            "original_language": self.original_language,
            "subtitles": {
                language: [track.to_dict() for track in tracks]
                for language, tracks in self.subtitles.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VideoMetadata":
        subtitles = {
            language: tuple(SubtitleTrack.from_dict(track) for track in tracks)
            for language, tracks in (value.get("subtitles") or {}).items()
        }
        return cls(
            video_id=value["video_id"],
            webpage_url=value["webpage_url"],
            title=value["title"],
            duration=value.get("duration"),
            uploader=value.get("uploader"),
            upload_date=value.get("upload_date"),
            language=value.get("language"),
            original_language=value.get("original_language"),
            subtitles=subtitles,
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
