import pytest

from ytdub.errors import SubtitleParseError
from ytdub.subtitles import TranscriptCue, normalize_cues, parse_subtitle_file, parse_subtitle_text


def test_vtt_parsing_and_normalization_produces_stable_segments() -> None:
    vtt = """WEBVTT

1
00:00:00.000 --> 00:00:01.500 align:start
<c.green>Hallo</c> &amp; willkommen

2
00:00:01.500 --> 00:00:03.000
Bonjour\nmonde

"""

    cues = parse_subtitle_text(vtt)
    segments = normalize_cues([*cues, cues[0]])

    assert [cue.text for cue in cues] == ["Hallo & willkommen", "Bonjour monde"]
    assert [segment.segment_id for segment in segments] == [
        "seg-000001-0000000000",
        "seg-000002-0000001500",
    ]
    assert [segment.target_duration for segment in segments] == [1.5, 1.5]


def test_subtitle_parser_rejects_missing_or_malformed_timestamps() -> None:
    with pytest.raises(SubtitleParseError, match="no non-empty timed cues"):
        parse_subtitle_text("WEBVTT\n\nplain text only\n")
    with pytest.raises(SubtitleParseError, match="malformed subtitle timestamp"):
        parse_subtitle_text("WEBVTT\n\n00:xx --> 00:02.000\nHallo\n")


def test_normalization_discards_invalid_cues_and_rejects_an_empty_result() -> None:
    with pytest.raises(SubtitleParseError, match="no usable"):
        normalize_cues([TranscriptCue(start=2.0, end=2.0, text="ignored")])


def test_json3_subtitles_are_accepted_when_vtt_is_not_available(tmp_path) -> None:
    subtitle = tmp_path / "manual.de.json3"
    subtitle.write_text(
        '{"events":[{"tStartMs":250,"dDurationMs":1250,"segs":[{"utf8":"Guten Tag"}]}]}',
        encoding="utf-8",
    )

    assert parse_subtitle_file(subtitle) == [TranscriptCue(0.25, 1.5, "Guten Tag")]
