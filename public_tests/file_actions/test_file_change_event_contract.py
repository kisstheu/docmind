from infra.file_change_store import FileChangeStore, interpret_file_event_type


def _snapshot(relative_path: str) -> dict:
    return {
        "relative_path": relative_path,
        "size": 16,
        "mtime": 1.0,
        "ctime": 1.0,
        "sha256": "0" * 64,
    }


def test_soft_delete_interpretation_accepts_current_and_legacy_markers() -> None:
    assert interpret_file_event_type("soft_delete") == "soft_delete"
    assert (
        interpret_file_event_type(
            "delete",
            reason="user_confirmed_soft_delete",
            after_path="archive/docs/alpha.md",
        )
        == "soft_delete"
    )
    assert (
        interpret_file_event_type(
            "delete",
            after_path=r".docmind_trash\session\docs\alpha.md",
        )
        == "soft_delete"
    )


def test_plain_delete_without_soft_delete_marker_is_preserved() -> None:
    assert (
        interpret_file_event_type(
            "delete",
            reason="synthetic_cleanup",
            after_path="archive/docs/alpha.md",
        )
        == "delete"
    )


def test_recorded_soft_delete_is_returned_with_explicit_type(tmp_path) -> None:
    store = FileChangeStore(tmp_path / "events.db")
    store.record_delete(
        notes_dir=tmp_path,
        before=_snapshot("docs/alpha.md"),
        after=_snapshot(".docmind_trash/session/docs/alpha.md"),
        reason="user_confirmed_soft_delete",
    )

    events = store.list_recent_events(notes_dir=tmp_path)

    assert len(events) == 1
    assert events[0]["event_type"] == "soft_delete"
    assert events[0]["before_path"] == "docs/alpha.md"
    assert events[0]["after_path"] == ".docmind_trash/session/docs/alpha.md"
