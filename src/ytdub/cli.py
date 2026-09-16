"""Command line entry point for the Milestone 1 vertical slice."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import time

from .errors import IngestionError
from .pipeline import IngestionConfig, IngestionPipeline
from .transcription import WhisperSettings


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without performing network or model work."""

    parser = argparse.ArgumentParser(
        prog="ytdub",
        description="Milestone 1: download a YouTube source and produce timestamped transcript segments.",
    )
    parser.add_argument("url", help="Single public YouTube video URL (not a playlist).")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"), help="Per-video cache root.")
    parser.add_argument(
        "--subtitle-language",
        help="Prefer this manually uploaded subtitle language (for example de or fr).",
    )
    parser.add_argument(
        "--source-language",
        help="Optional source-language hint for subtitle selection and faster-whisper (for example hi).",
    )
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model for fallback ASR.")
    parser.add_argument("--whisper-device", default="auto", help="faster-whisper device, default: auto.")
    parser.add_argument(
        "--whisper-compute-type",
        default="int8",
        help="faster-whisper compute type, default: int8.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="rerun Whisper on the cached media and print raw ASR diagnostics without caching",
    )
    return parser


def configure_logging() -> None:
    """Use concise terminal progress messages for an interactive CLI run."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    """Run Milestone 1 and return a shell-friendly exit status."""

    args = build_parser().parse_args(argv)
    configure_logging()
    started = time.perf_counter()
    logging.info("Milestone 1 started: ingestion and transcription only")
    config = IngestionConfig(
        url=args.url,
        cache_root=args.cache_dir,
        source_language=args.source_language,
        subtitle_language=args.subtitle_language,
        whisper=WhisperSettings(
            model=args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        ),
        diagnostic=args.diagnostic,
    )
    try:
        result = IngestionPipeline().run(config)
    except IngestionError as error:
        logging.error("Milestone 1 failed: %s", error)
        return 1
    except Exception:
        logging.exception("Milestone 1 failed unexpectedly")
        return 1

    elapsed = time.perf_counter() - started
    if args.diagnostic:
        audio_duration = _probe_audio_duration(result.media.audio_path)
        logging.info("Diagnostic audio_path=%s", result.media.audio_path)
        logging.info("Diagnostic audio_duration=%.3fs", audio_duration)
        logging.info("Diagnostic whisper=%s", config.whisper.to_dict())
        logging.info(
            "Diagnostic detected_language=%s probability=%s",
            result.transcript.language or "unknown",
            result.transcript.language_probability,
        )
        for index, cue in enumerate(result.transcript.raw_cues, start=1):
            logging.info(
                "Diagnostic raw_segment=%d start=%.3f end=%.3f text=%r",
                index,
                cue.start,
                cue.end,
                cue.text,
            )
    logging.info(
        "Milestone 1 complete: provider=%s language=%s segments=%d",
        result.transcript.provider,
        result.transcript.language or "unknown",
        len(result.segments),
    )
    logging.info("Artifacts: %s", result.cache_directory)
    logging.info("Total processing time: %.2fs", elapsed)
    return 0


def _probe_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


if __name__ == "__main__":
    sys.exit(main())
