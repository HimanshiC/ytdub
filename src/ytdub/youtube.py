"""yt-dlp metadata probing and separate video/audio artifact acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from .cache import VideoCache, cache_key
from .errors import IngestionError
from .models import VideoMetadata
from .subtitles import _import_yt_dlp


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MediaArtifacts:
    """Paths to independently downloaded source video and audio streams."""

    video_path: Path
    audio_path: Path

    def to_dict(self) -> dict[str, str]:
        return {"video_path": str(self.video_path), "audio_path": str(self.audio_path)}

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> "MediaArtifacts":
        return cls(video_path=Path(value["video_path"]), audio_path=Path(value["audio_path"]))


class YouTubeIngestor:
    """Small deterministic facade over yt-dlp for this milestone."""

    def probe(self, url: str) -> VideoMetadata:
        """Fetch metadata and list manually uploaded subtitle tracks."""

        yt_dlp = _import_yt_dlp()
        LOGGER.info("Ingestion: fetching YouTube metadata")
        options = {"noplaylist": True, "quiet": True, "no_warnings": True}
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=False)
        except Exception as error:
            raise IngestionError(f"could not fetch YouTube metadata: {error}") from error
        if not isinstance(info, dict):
            raise IngestionError("yt-dlp returned playlist/unsupported metadata instead of one video")
        video = VideoMetadata.from_yt_dlp(info, url)
        LOGGER.info(
            "Ingestion: video id=%s duration=%s manual_subtitle_languages=%s",
            video.video_id,
            _format_duration(video.duration),
            ", ".join(sorted(video.subtitles)) or "none",
        )
        return video

    def acquire_media(self, video: VideoMetadata, cache: VideoCache) -> MediaArtifacts:
        """Reuse or download independent best video-only and audio-only source files."""

        yt_dlp = _import_yt_dlp()
        media_key = cache_key(
            {
                "url": video.webpage_url,
                "video_id": video.video_id,
                "duration": video.duration,
                "strategy": "bestvideo[ext=mp4]/bestvideo + bestaudio/best separate streams",
                "yt_dlp_version": getattr(yt_dlp.version, "__version__", "unknown"),
            }
        )
        cached = cache.read_json("media.json", media_key)
        if isinstance(cached, dict):
            artifacts = MediaArtifacts.from_dict(cached)
            if artifacts.video_path.is_file() and artifacts.audio_path.is_file():
                LOGGER.info("Ingestion: reusing cached separate media artifacts")
                return artifacts

        media_directory = cache.artifact_path("media/.keep").parent
        LOGGER.info("Ingestion: downloading video-only source stream")
        video_path = self._download_stream(
            yt_dlp=yt_dlp,
            video=video,
            output_template=media_directory / "source_video.%(ext)s",
            format_selector="bestvideo[ext=mp4]/bestvideo",
            kind="video",
        )
        LOGGER.info("Ingestion: downloading audio-only source stream")
        audio_path = self._download_stream(
            yt_dlp=yt_dlp,
            video=video,
            output_template=media_directory / "source_audio.%(ext)s",
            format_selector="bestaudio/best",
            kind="audio",
        )
        artifacts = MediaArtifacts(video_path=video_path, audio_path=audio_path)
        cache.write_json("media.json", media_key, artifacts.to_dict())
        LOGGER.info("Ingestion: separate source media artifacts are ready")
        return artifacts

    def _download_stream(
        self,
        yt_dlp: Any,
        video: VideoMetadata,
        output_template: Path,
        format_selector: str,
        kind: str,
    ) -> Path:
        output_template.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "format": format_selector,
            "outtmpl": {"default": str(output_template)},
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.extract_info(video.webpage_url, download=True)
        except Exception as error:
            raise IngestionError(f"could not download {kind}-only stream: {error}") from error

        output_prefix = output_template.name.split(".%(ext)s", maxsplit=1)[0]
        created_files = sorted(
            path
            for path in output_template.parent.iterdir()
            if path.is_file()
            and not path.name.endswith((".part", ".ytdl"))
            and path.name.startswith(output_prefix)
        )
        if not created_files:
            raise IngestionError(f"yt-dlp did not create a {kind}-only source artifact")
        if len(created_files) > 1:
            raise IngestionError(
                f"yt-dlp created multiple {kind} artifacts; refusing ambiguous output: {created_files}"
            )
        return created_files[0]


def _format_duration(duration: float | None) -> str:
    return f"{duration:.1f}s" if duration is not None else "unknown"
