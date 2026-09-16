from ytdub.transcription import TimedWord, refine_timed_words
from ytdub.subtitles import TranscriptCue


def test_word_timestamps_split_an_80_second_punctuation_free_recording() -> None:
    words = tuple(
        TimedWord(start=float(index), end=float(index) + 0.4, text=f" word{index}")
        for index in range(83)
    )

    cues = refine_timed_words(words)

    assert len(cues) > 1
    assert cues[0].start == 0.0
    assert cues[-1].end == 82.4
    assert all(cue.end - cue.start <= 12.0 for cue in cues)
    assert sum(cue.text.count("word") for cue in cues) == 83


def test_word_refinement_prefers_punctuation_and_observed_silence() -> None:
    words = (
        TimedWord(1.0, 1.3, " Guten"),
        TimedWord(1.3, 1.7, " Tag."),
        TimedWord(3.0, 3.4, " Wie"),
        TimedWord(3.4, 3.8, " geht"),
        TimedWord(3.8, 4.2, " es"),
        TimedWord(4.2, 4.5, " dir"),
    )

    cues = refine_timed_words(words)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (1.0, 1.7, "Guten Tag."),
        (3.0, 4.5, "Wie geht es dir"),
    ]


def test_word_refinement_does_not_treat_an_initial_as_a_sentence_boundary() -> None:
    words = (
        TimedWord(15.8, 16.04, " V."),
        TimedWord(16.12, 16.4, " und"),
        TimedWord(16.45, 16.9, " 30."),
    )

    assert refine_timed_words(words) == [TranscriptCue(15.8, 16.9, "V. und 30.")]
