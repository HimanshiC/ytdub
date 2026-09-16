from ytdub.cache import VideoCache, cache_key, file_sha256


def test_cache_key_is_order_independent_and_json_artifacts_are_keyed(tmp_path) -> None:
    first_key = cache_key({"a": 1, "b": ["x", "y"]})
    second_key = cache_key({"b": ["x", "y"], "a": 1})
    assert first_key == second_key

    cache = VideoCache(tmp_path, "video-id")
    cache.write_json("metadata.json", first_key, {"title": "Test"})

    assert cache.read_json("metadata.json", first_key) == {"title": "Test"}
    assert cache.read_json("metadata.json", "changed-input") is None


def test_file_sha256_hashes_streamed_artifact_content(tmp_path) -> None:
    artifact = tmp_path / "audio.webm"
    artifact.write_bytes(b"source audio bytes")

    first_hash = file_sha256(artifact)
    assert first_hash == file_sha256(artifact)
    artifact.write_bytes(b"different source audio bytes")
    assert file_sha256(artifact) != first_hash
