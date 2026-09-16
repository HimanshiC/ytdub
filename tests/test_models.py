import pytest

from ytdub.models import Segment


def test_segment_derives_target_duration_and_round_trips() -> None:
    segment = Segment(
        segment_id="seg-000001-0000000000",
        start=0.0,
        end=2.5,
        source_text="Hallo zusammen",
    )

    assert segment.target_duration == 2.5
    assert Segment.from_dict(segment.to_dict()) == segment


def test_segment_rejects_a_duration_that_changes_source_timing() -> None:
    with pytest.raises(ValueError, match="target_duration"):
        Segment(
            segment_id="seg-000001-0000000000",
            start=1.0,
            end=3.0,
            source_text="Bonjour",
            target_duration=1.5,
        )
