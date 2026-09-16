"""Baseline translation, TTS, timestamped audio assembly, and video muxing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Iterable

import edge_tts
# from deep_translator import GoogleTranslator
from .translator import GroqTranslator
from .cache import VideoCache, cache_key
from .errors import RuntimeDependencyError
from .models import Segment, VideoMetadata
from .youtube import MediaArtifacts


LOGGER = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US-AriaNeural"
# TRANSLATION_BACKEND = "googletrans-deep-translator-v1"
TRANSLATION_BACKEND = "groq-gpt-oss-20b-v1"
TTS_BACKEND = "edge-tts-v1"
MAX_TTS_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class DubbingConfig:
    voice: str = DEFAULT_VOICE
    tts_concurrency: int = MAX_TTS_CONCURRENCY


@dataclass(frozen=True, slots=True)
class DubbingResult:
    segments: tuple[Segment, ...]
    audio_path: Path
    video_path: Path
    output_path: Path
    duration: float


class BaselineDubbingPipeline:
    """Straightforward per-segment baseline used before adaptive synchronization."""

    def __init__(self, config: DubbingConfig | None = None) -> None:
        self.config = config or DubbingConfig()

    def run(
        self,
        *,
        video: VideoMetadata,
        media: MediaArtifacts,
        segments: Iterable[Segment],
        cache: VideoCache,
    ) -> DubbingResult:
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("cannot dub an empty segment list")

        translated = self._translate_segments(video, segment_list, cache)
        spoken = self._synthesize_segments(translated, cache)

        audio_path = self._assemble_audio(
            video=video,
            segments=spoken,
            cache=cache,
        )

        output_path = cache.artifact_path("output/dubbed_baseline.mp4")
        self._mux_video(
            video_path=media.video_path,
            audio_path=audio_path,
            output_path=output_path,
        )

        duration = _probe_duration(output_path)

        # Persist the complete baseline segment state for later milestones.
        key = cache_key(
            {
                "pipeline": "milestone-2-baseline",
                "video_id": video.video_id,
                "voice": self.config.voice,
                "translation_backend": TRANSLATION_BACKEND,
                "tts_backend": TTS_BACKEND,
                "segments": [segment.to_dict() for segment in spoken],
            }
        )
        cache.write_json(
            "dub_segments.json",
            key,
            [segment.to_dict() for segment in spoken],
        )

        return DubbingResult(
            segments=tuple(spoken),
            audio_path=audio_path,
            video_path=media.video_path,
            output_path=output_path,
            duration=duration,
        )
    def _translate_segments(
        self,
        video: VideoMetadata,
        segments: list[Segment],
        cache: VideoCache,
    ) -> list[Segment]:
        LOGGER.info("Translation: translating %d segments", len(segments))

        translator = GroqTranslator()

        result = list(segments)

        source_language = (
            video.language
            or video.original_language
            or "auto"
        ).split("-")[0].lower()

        for index, segment in enumerate(segments):
            key = cache_key(
                {
                    "backend": TRANSLATION_BACKEND,
                    "source_language": source_language,
                    "target_language": "en",
                    "mode": "baseline",
                    "source_text": segment.source_text,
                }
            )

            cached = cache.read_json(
                f"translation/{segment.segment_id}.json",
                key,
            )

            if (
                isinstance(cached, dict)
                and isinstance(cached.get("translated_text"), str)
                and cached["translated_text"].strip()
            ):
                translated_text = cached["translated_text"].strip()

                LOGGER.info(
                    "Translation [%d/%d] reused %s",
                    index + 1,
                    len(segments),
                    segment.segment_id,
                )
            else:
                LOGGER.info(
                    "Translation [%d/%d] Groq baseline %s",
                    index + 1,
                    len(segments),
                    segment.segment_id,
                )

                translated_text = translator.translate_segment(
                    segment,
                )

                cache.write_json(
                    f"translation/{segment.segment_id}.json",
                    key,
                    {
                        "segment_id": segment.segment_id,
                        "source_language": source_language,
                        "source_text": segment.source_text,
                        "translated_text": translated_text,
                        "mode": "baseline",
                    },
                )

            result[index].translated_text = translated_text

            LOGGER.info(
                "Translation [%d/%d] %s -> %s",
                index + 1,
                len(result),
                segment.source_text,
                translated_text,
            )

        return result
    #older implementation of translation using GoogleTranslator, now commented out
    # def _translate_segments(
    #     self,
    #     video: VideoMetadata,
    #     segments: list[Segment],
    #     cache: VideoCache,
    # ) -> list[Segment]:
    #     LOGGER.info("Translation: translating %d segments", len(segments))

    #     source_language = (
    #         video.language
    #         or video.original_language
    #         or "auto"
    #     ).split("-")[0].lower()

    #     pending: list[tuple[int, Segment]] = []
    #     result = list(segments)

    #     # First reuse anything already cached.
    #     for index, segment in enumerate(segments):
    #         key = cache_key(
    #             {
    #                 "backend": TRANSLATION_BACKEND,
    #                 "source_language": source_language,
    #                 "target_language": "en",
    #                 "source_text": segment.source_text,
    #             }
    #         )

    #         cached = cache.read_json(
    #             f"translation/{segment.segment_id}.json",
    #             key,
    #         )

    #         if (
    #             isinstance(cached, dict)
    #             and isinstance(cached.get("translated_text"), str)
    #             and cached["translated_text"].strip()
    #         ):
    #             result[index].translated_text = cached["translated_text"].strip()
    #         else:
    #             pending.append((index, segment))

    #     if not pending:
    #         LOGGER.info("Translation: all segments reused from cache")
    #         return result

    #     translator = GoogleTranslator(
    #         source=source_language,
    #         target="en",
    #     )

    #     texts = [segment.source_text for _, segment in pending]

    #     LOGGER.info(
    #         "Translation: batch translating %d uncached segments",
    #         len(texts),
    #     )

    #     translated_batch: list[str | None] = []

    #     for attempt in range(3):
    #         try:
    #             translated_batch = translator.translate_batch(texts)
    #             break
    #         except Exception as error:
    #             if attempt == 2:
    #                 raise RuntimeError(
    #                     f"batch translation failed after 3 attempts: {error}"
    #                 ) from error

    #             wait_seconds = 2 ** attempt
    #             LOGGER.warning(
    #                 "Translation: batch request failed, retrying in %ds: %s",
    #                 wait_seconds,
    #                 error,
    #             )
    #             import time
    #             time.sleep(wait_seconds)

    #     if len(translated_batch) != len(pending):
    #         raise RuntimeError(
    #             "translation backend returned an unexpected number of results: "
    #             f"expected {len(pending)}, got {len(translated_batch)}"
    #         )

    #     for (index, segment), translated_text in zip(
    #         pending,
    #         translated_batch,
    #         strict=True,
    #     ):
    #         if not translated_text or not translated_text.strip():
    #             raise RuntimeError(
    #                 f"translation returned empty text for {segment.segment_id}"
    #             )

    #         translated_text = translated_text.strip()
    #         result[index].translated_text = translated_text

    #         key = cache_key(
    #             {
    #                 "backend": TRANSLATION_BACKEND,
    #                 "source_language": source_language,
    #                 "target_language": "en",
    #                 "source_text": segment.source_text,
    #             }
    #         )

    #         cache.write_json(
    #             f"translation/{segment.segment_id}.json",
    #             key,
    #             {
    #                 "segment_id": segment.segment_id,
    #                 "source_text": segment.source_text,
    #                 "translated_text": translated_text,
    #             },
    #         )

    #     for index, segment in enumerate(result, start=1):
    #         LOGGER.info(
    #             "Translation [%d/%d] %s -> %s",
    #             index,
    #             len(result),
    #             segment.source_text,
    #             segment.translated_text,
    #         )

    #     return result

    def _synthesize_segments(
        self,
        segments: list[Segment],
        cache: VideoCache,
    ) -> list[Segment]:
        LOGGER.info(
            "TTS: synthesizing %d segments with voice=%s concurrency=%d",
            len(segments),
            self.config.voice,
            self.config.tts_concurrency,
        )

        async def run_all() -> None:
            semaphore = asyncio.Semaphore(max(1, self.config.tts_concurrency))

            async def one(segment: Segment, index: int) -> None:
                if not segment.translated_text:
                    raise ValueError(f"segment {segment.segment_id} has no translated text")

                key = cache_key(
                    {
                        "backend": TTS_BACKEND,
                        "voice": self.config.voice,
                        "text": segment.translated_text,
                    }
                )

                cached = cache.read_json(
                    f"tts/{segment.segment_id}.json",
                    key,
                )

                if isinstance(cached, dict):
                    path_value = cached.get("artifact_path")
                    duration_value = cached.get("duration")
                    if (
                        isinstance(path_value, str)
                        and isinstance(duration_value, (int, float))
                        and Path(path_value).is_file()
                    ):
                        segment.tts_artifact_path = path_value
                        segment.actual_tts_duration = float(duration_value)
                        LOGGER.info(
                            "TTS [%d/%d] reused %s",
                            index,
                            len(segments),
                            segment.segment_id,
                        )
                        return

                async with semaphore:
                    output_path = cache.artifact_path(
                        f"tts/{segment.segment_id}.mp3"
                    )
                    communicate = edge_tts.Communicate(
                        segment.translated_text,
                        self.config.voice,
                    )
                    await communicate.save(str(output_path))

                duration = _probe_duration(output_path)

                segment.tts_artifact_path = str(output_path)
                segment.actual_tts_duration = duration

                cache.write_json(
                    f"tts/{segment.segment_id}.json",
                    key,
                    {
                        "segment_id": segment.segment_id,
                        "artifact_path": str(output_path),
                        "duration": duration,
                    },
                )

                LOGGER.info(
                    "TTS [%d/%d] %s duration=%.3fs target=%.3fs",
                    index,
                    len(segments),
                    segment.segment_id,
                    duration,
                    segment.target_duration or 0.0,
                )

            await asyncio.gather(
                *(one(segment, index) for index, segment in enumerate(segments, start=1))
            )

        asyncio.run(run_all())

        for segment in segments:
            target = segment.target_duration or 0.0
            actual = segment.actual_tts_duration or 0.0

            if actual > target + 0.05:
                segment.synchronization_status = "OVERFLOW"
                segment.synchronization_action = (
                    f"BASELINE_OVERFLOW:+{actual - target:.3f}s"
                )
                LOGGER.warning(
                    "SYNC baseline overflow %s actual=%.3fs target=%.3fs",
                    segment.segment_id,
                    actual,
                    target,
                )
            else:
                segment.synchronization_status = "PLACED"
                segment.synchronization_action = "ABSOLUTE_START"

        return segments

    def _assemble_audio(
        self,
        *,
        video: VideoMetadata,
        segments: list[Segment],
        cache: VideoCache,
    ) -> Path:
        output_path = cache.artifact_path("output/dubbed_audio.m4a")

        duration = (
            float(video.duration)
            if video.duration is not None
            else max(
                segment.end
                for segment in segments
            )
        )

        ffmpeg_inputs: list[str] = [
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            "anullsrc=r=48000:cl=stereo",
        ]

        filters: list[str] = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[base]"]

        mix_labels = ["[base]"]

        for index, segment in enumerate(segments, start=1):
            if not segment.tts_artifact_path:
                continue

            ffmpeg_inputs.extend(["-i", segment.tts_artifact_path])

            delay_ms = max(0, round(segment.start * 1000))

            filters.append(
                f"[{index}:a]"
                f"aformat=sample_rates=48000:channel_layouts=stereo,"
                f"adelay={delay_ms}|{delay_ms}"
                f"[seg{index}]"
            )
            mix_labels.append(f"[seg{index}]")

        if len(mix_labels) == 1:
            filter_complex = "[base]anull"
        else:
            filter_complex = (
                "".join(mix_labels)
                + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,"
                f"atrim=duration={duration:.3f},asetpts=N/SR/TB[out]"
            )

        command = [
            "ffmpeg",
            "-y",
            *ffmpeg_inputs,
            "-filter_complex",
            ";".join(filters + [filter_complex]),
            "-map",
            "[out]" if len(mix_labels) > 1 else "[base]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.3f}",
            str(output_path),
        ]

        LOGGER.info("Audio: assembling timestamped baseline dub")
        _run_ffmpeg(command)

        LOGGER.info(
            "Audio: baseline dub ready duration=%.3fs path=%s",
            _probe_duration(output_path),
            output_path,
        )
        return output_path

    def _mux_video(
        self,
        *,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        LOGGER.info("Mux: replacing original audio and copying video stream")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

        _run_ffmpeg(command)

        LOGGER.info("Mux: output ready at %s", output_path)


def _translate_text(text: str, *, source_language: str | None) -> str:
    source = (source_language or "auto").split("-")[0].lower()

    try:
        translator = GoogleTranslator(
            source=source,
            target="en",
        )
        translated = translator.translate(text)
    except Exception as error:
        raise RuntimeError(
            f"translation failed for {text!r}: {error}"
        ) from error

    if not translated or not translated.strip():
        raise RuntimeError(f"translation returned empty text for {text!r}")

    return translated.strip()


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeDependencyError(
            "ffprobe is required for duration measurement."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffprobe failed for {path}: {error.stderr.strip()}"
        ) from error

    return float(result.stdout.strip())


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeDependencyError(
            "ffmpeg is required for audio assembly and video muxing."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "FFmpeg failed:\n"
            f"{error.stderr[-4000:]}"
        ) from error