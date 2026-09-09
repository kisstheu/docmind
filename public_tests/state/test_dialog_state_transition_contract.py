import pytest

from ai.table_presentation import StructuredTable, TableRenderOptions
from app.dialog_state_machine import ConversationState, DialogEvent, apply_event_to_state


@pytest.mark.parametrize(
    ("event", "expected_mode"),
    [
        (DialogEvent(name="metadata", route_hint="repo_meta"), "repo_meta"),
        (DialogEvent(name="content", route_hint="normal_retrieval"), "content"),
        (DialogEvent(name="smalltalk"), "smalltalk"),
    ],
)
def test_event_application_sets_mode_without_mutating_input(event, expected_mode) -> None:
    original = ConversationState(mode="idle", last_user_question="Synthetic question.")

    transitioned = apply_event_to_state(original, event)

    assert transitioned.mode == expected_mode
    assert original.mode == "idle"


def test_event_application_preserves_control_state_fields() -> None:
    presentation_table = StructuredTable(
        columns=("对象",),
        rows=(("合成项甲",),),
    )
    original = ConversationState(
        last_factual_answer_text="合成项甲已有说明。",
        current_presentation_table=presentation_table,
        current_presentation_options=TableRenderOptions(compact=True),
        last_answer_strategy="direct_evidence",
        last_answer_source_files=["docs/alpha.md"],
        last_result_set_items=["docs/alpha.md", "docs/beta.md"],
        last_result_set_entity_type="file",
        last_selected_candidate="Candidate A",
        last_selected_source_files=["docs/alpha.md"],
        pending_action_type="rename",
        pending_action_source_path="docs/alpha.md",
        pending_action_target_path="docs/gamma.md",
    )

    transitioned = apply_event_to_state(
        original,
        DialogEvent(name="content", route_hint="normal_retrieval"),
    )

    assert transitioned.last_result_set_items == ["docs/alpha.md", "docs/beta.md"]
    assert transitioned.last_factual_answer_text == "合成项甲已有说明。"
    assert transitioned.current_presentation_table == presentation_table
    assert transitioned.current_presentation_options == TableRenderOptions(compact=True)
    assert transitioned.last_answer_strategy == "direct_evidence"
    assert transitioned.last_answer_source_files == ["docs/alpha.md"]
    assert transitioned.last_selected_candidate == "Candidate A"
    assert transitioned.last_selected_source_files == ["docs/alpha.md"]
    assert transitioned.pending_action_type == "rename"
    assert transitioned.pending_action_source_path == "docs/alpha.md"
    assert transitioned.pending_action_target_path == "docs/gamma.md"
