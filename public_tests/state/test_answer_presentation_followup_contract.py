from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.table_presentation import StructuredTable, TableRenderOptions
from app.chat_state_helpers import update_state_after_answer_presentation
from app.dialog import task_semantics
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.dialog.task_semantics import is_answer_presentation_followup


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None


def _answer_entity_state(answer: str) -> ConversationState:
    return ConversationState(
        last_user_question="上一轮问题",
        last_route="normal_retrieval",
        last_content_user_question="上一轮问题",
        last_content_route="normal_retrieval",
        last_effective_search_query="合成检索主题",
        last_answer_text=answer,
        last_answer_preview=answer[:200],
        last_answer_type="enumeration_generated",
        last_result_set_items=["合成项甲", "合成项乙"],
        last_result_set_entity_type="对象",
        last_generated_result_items=["合成项甲", "合成项乙"],
        last_generated_result_entity_type="对象",
        last_generated_result_source_candidates=["资料甲.md", "资料乙.md"],
        last_generated_result_source_hits=[["资料甲.md"], ["资料乙.md"]],
        last_generated_result_focuses=["focus-a", "focus-b"],
    )


@pytest.mark.parametrize(
    ("question", "expected_clauses", "expected_presentation", "expected_new_fact"),
    [
        ("合并成表格", ["合并成表格"], True, False),
        ("把前面的回答合并成表格", ["把前面的回答合并成表格"], True, False),
        ("把这些内容合并整理成表格", ["把这些内容合并整理成表格"], True, False),
        ("并列整理成表格", ["并列整理成表格"], True, False),
        ("再整理成表格", ["再整理成表格"], True, False),
        ("把上面的内容重新整理成列表", ["把上面的内容重新整理成列表"], True, False),
        ("做成表格并补充价格", ["做成表格", "补充价格"], False, True),
        ("整理成表格且标出出处", ["整理成表格", "标出出处"], False, True),
        ("做成表格再看看18岁以下", ["做成表格", "看看18岁以下"], False, True),
        ("换成列表并告诉我截止时间", ["换成列表", "告诉我截止时间"], False, True),
    ],
)
def test_presentation_clause_segmentation_is_lexically_safe_and_demand_aware(
    question,
    expected_clauses,
    expected_presentation,
    expected_new_fact,
):
    clauses = task_semantics._split_followup_clauses(question)
    clause_intents = [
        task_semantics._has_answer_presentation_intent(clause)
        for clause in clauses
    ]
    observation = {
        "question": question,
        "clauses": clauses,
        "presentation_intent": is_answer_presentation_followup(
            question,
            has_previous_answer=True,
        ),
        "new_factual_demand": any(not intent for intent in clause_intents),
    }

    assert observation == {
        "question": question,
        "clauses": expected_clauses,
        "presentation_intent": expected_presentation,
        "new_factual_demand": expected_new_fact,
    }


@pytest.mark.parametrize(
    "prefix",
    ["", "请", "帮我", "给我", "那", "那么", "那请", "可以帮我"],
)
@pytest.mark.parametrize("suffix", ["", "一下", "看看", "吧", "吗", "呢"])
def test_soft_modifiers_preserve_pure_presentation_semantics(prefix, suffix):
    question = f"{prefix}做成表格{suffix}"

    assert is_answer_presentation_followup(
        question,
        has_previous_answer=True,
    ) is True


@pytest.mark.parametrize(
    ("answer", "question"),
    [
        ("岗位甲负责接口，岗位乙负责数据处理。", "整理成表格"),
        ("条款甲约定交付；条款乙约定验收。", "换成要点"),
        ("候选项甲稳定，候选项乙灵活。", "按三列整理"),
        ("课程甲讲基础，课程乙讲实践。", "简短一点"),
    ],
)
def test_pure_cross_domain_presentation_transform_precedes_answer_entity_scope(
    answer,
    question,
):
    state = _answer_entity_state(answer)

    event = detect_dialog_event(question, state, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name == "answer_presentation_followup"
    assert event.route_hint is None
    assert scope.answer_entity_followup is False
    assert scope.result_scope_paths is None


@pytest.mark.parametrize(
    "question",
    [
        "可以给我个表格吗？",
        "做成表格",
        "换成列表",
        "简短一点",
        "按类别排一下",
        "做成三列",
        "用 Markdown 表格",
        "换一种格式",
        "用要点表示",
    ],
)
def test_domain_neutral_pure_presentation_variants_are_answer_transforms(question):
    state = _answer_entity_state("合成项甲与合成项乙已有完整说明。")

    event = detect_dialog_event(question, state, _LoggerStub())

    assert event.name == "answer_presentation_followup"


@pytest.mark.parametrize(
    ("answer", "question"),
    [
        ("岗位甲与岗位乙已有职责说明。", "给我个表格吧"),
        ("条款甲与条款乙已有期限说明。", "给我做个表格"),
        ("采购项甲与采购项乙已有条件说明。", "弄成表格吧"),
        ("课程甲与课程乙已有课时说明。", "表格形式呢"),
        ("岗位甲与岗位乙已有职责说明。", "那做成表格"),
        ("条款甲与条款乙已有期限说明。", "换个表格看看"),
        ("采购项甲与采购项乙已有条件说明。", "用表格表示一下"),
        ("课程甲与课程乙已有课时说明。", "帮我整理成表格"),
    ],
)
def test_natural_cross_domain_presentation_grammar_uses_previous_answer_only(
    answer,
    question,
):
    state = _answer_entity_state(answer)

    event = detect_dialog_event(question, state, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name == "answer_presentation_followup"
    assert event.route_hint is None
    assert scope.answer_entity_followup is False
    assert scope.requires_result_set_generation is False
    assert scope.result_scope_paths is None


@pytest.mark.parametrize(
    ("answer", "question"),
    [
        ("采购项甲与采购项乙已有规格说明。", "给我个表格，再补上价格"),
        ("条款甲与条款乙已有期限说明。", "做成表格并标出出处"),
        ("岗位甲与岗位乙已有职责说明。", "表格整理一下，再看看18岁以下"),
        ("课程甲与课程乙已有课时说明。", "做表格并补充每项截止时间"),
        ("对象甲与对象乙已有特征说明。", "做成表格，再告诉我哪个最严重"),
    ],
)
def test_format_request_with_a_new_fact_demand_keeps_retrieval_path(answer, question):
    state = _answer_entity_state(answer)

    event = detect_dialog_event(question, state, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name != "answer_presentation_followup"
    assert scope.answer_entity_followup is True
    assert scope.result_scope_paths == ("资料甲.md", "资料乙.md")


@pytest.mark.parametrize(
    "question",
    [
        "做成表格并按价格排序",
        "按价格做表格",
    ],
)
def test_factual_sort_or_axis_is_not_reduced_to_a_presentation_transform(question):
    state = _answer_entity_state("上一轮只包含已有摘要。")

    event = detect_dialog_event(question, state, _LoggerStub())

    assert event.name != "answer_presentation_followup"
    assert event.route_hint == "normal_retrieval"


def test_presentation_wording_without_a_previous_answer_is_not_a_followup():
    assert is_answer_presentation_followup(
        "整理成表格",
        has_previous_answer=False,
    ) is False


def test_independent_topic_with_a_format_request_is_not_a_previous_answer_transform():
    state = _answer_entity_state("上一轮是无关的合成摘要。")

    event = detect_dialog_event("请把采购风险做个表", state, _LoggerStub())

    assert event.name != "answer_presentation_followup"
    assert event.route_hint == "normal_retrieval"


def test_presentation_state_update_preserves_content_and_result_set_authority():
    state = _answer_entity_state("原回答")
    before = SimpleNamespace(
        content_question=state.last_content_user_question,
        search_query=state.last_effective_search_query,
        result_items=list(state.last_result_set_items or []),
        generated_items=list(state.last_generated_result_items or []),
        source_candidates=list(state.last_generated_result_source_candidates or []),
        source_hits=[list(hits) for hits in state.last_generated_result_source_hits or []],
        focuses=list(state.last_generated_result_focuses or []),
        answer_type=state.last_answer_type,
    )

    updated = update_state_after_answer_presentation(
        state,
        "换成列表",
        "- 合成项甲\n- 合成项乙",
    )

    assert updated.last_answer_text == "- 合成项甲\n- 合成项乙"
    assert updated.last_factual_answer_text == "原回答"
    assert updated.current_presentation_table is None
    assert updated.last_answer_strategy == "last_answer_transform"
    assert updated.last_content_user_question == before.content_question
    assert updated.last_effective_search_query == before.search_query
    assert updated.last_result_set_items == before.result_items
    assert updated.last_generated_result_items == before.generated_items
    assert updated.last_generated_result_source_candidates == before.source_candidates
    assert updated.last_generated_result_source_hits == before.source_hits
    assert updated.last_generated_result_focuses == before.focuses
    assert updated.last_answer_type == before.answer_type


def test_structured_presentation_state_keeps_factual_authority_separate_from_view():
    state = _answer_entity_state("对象甲已有说明，对象乙已有说明。")
    table = StructuredTable(
        columns=("对象", "说明"),
        rows=(("对象甲", "已有说明"), ("对象乙", "已有说明")),
    )
    options = TableRenderOptions(balanced=True)

    updated = update_state_after_answer_presentation(
        state,
        "给我个表格",
        "本地渲染表格",
        table=table,
        options=options,
    )
    updated = update_state_after_answer_presentation(
        updated,
        "可以整齐一点吗？",
        "本地重渲染表格",
        table=table,
        options=options,
    )

    assert updated.last_factual_answer_text == "对象甲已有说明，对象乙已有说明。"
    assert updated.last_answer_text == "本地重渲染表格"
    assert updated.current_presentation_table == table
    assert updated.current_presentation_options == options


@pytest.mark.parametrize(
    ("columns", "rows", "question"),
    [
        (("候选人", "经验"), (("候选人X", "三年"),), "可以整齐一点吗？"),
        (("条款", "期限"), (("条款甲", "30天"),), "按期限排序"),
        (("采购项", "包装"), (("采购项甲", "密封"),), "把包装列放到最前面"),
    ],
)
def test_current_table_refinement_routes_before_content_followup_across_domains(
    columns,
    rows,
    question,
):
    state = _answer_entity_state("已有事实回答。")
    state.current_presentation_table = StructuredTable(columns=columns, rows=rows)
    state.current_presentation_options = TableRenderOptions()

    event = detect_dialog_event(question, state, _LoggerStub())

    assert event.name == "answer_presentation_followup"
    assert event.route_hint is None


@pytest.mark.parametrize(
    ("columns", "rows", "question"),
    [
        (
            ("候选人", "文件名"),
            (("候选人X", "10_简历.pdf"), ("候选人Y", "2_简历.pdf")),
            "按文件名排一下",
        ),
        (
            ("条款", "版本"),
            (("条款甲", "第12版"), ("条款乙", "第3版")),
            "按版本排",
        ),
        (
            ("采购项", "批次"),
            (("采购项甲", "批次2"), ("采购项乙", "批次11")),
            "按批次倒序排吧",
        ),
    ],
)
def test_existing_column_short_sort_routes_as_pure_table_refinement(
    columns,
    rows,
    question,
):
    state = _answer_entity_state("已有事实回答。")
    state.current_presentation_table = StructuredTable(columns=columns, rows=rows)

    event = detect_dialog_event(question, state, _LoggerStub())

    assert event.name == "answer_presentation_followup"
    assert event.route_hint is None


@pytest.mark.parametrize(
    "question",
    [
        "整齐一点并补充薪资",
        "按价格排序",
        "按价格排一下",
        "按说明排一下并告诉我出处",
        "增加风险列",
    ],
)
def test_current_table_refinement_does_not_capture_adjacent_new_fact_demands(question):
    state = _answer_entity_state("已有事实回答。")
    state.current_presentation_table = StructuredTable(
        columns=("对象", "说明"),
        rows=(("合成项甲", "已有说明"),),
    )

    event = detect_dialog_event(question, state, _LoggerStub())

    assert event.name != "answer_presentation_followup"
    assert event.route_hint == "normal_retrieval"


@pytest.mark.parametrize("upstream_route", ["answer_entity", "content", "result_set"])
def test_pure_presentation_precedes_all_retrieval_followup_families(upstream_route):
    state = _answer_entity_state("上一轮已有完整事实回答。")
    if upstream_route == "content":
        state.last_generated_result_items = None
        state.last_generated_result_entity_type = None
        state.last_generated_result_source_candidates = None
        state.last_result_set_focus_file = "资料甲.md"
    elif upstream_route == "result_set":
        state.last_generated_result_items = None
        state.last_generated_result_entity_type = None
        state.last_generated_result_source_candidates = None
        state.last_result_set_items = ["资料甲.md", "资料乙.md"]
        state.last_result_set_entity_type = "文件"
        state.last_result_set_selectable = True

    event = detect_dialog_event("表格形式呢", state, _LoggerStub())
    signals = analyze_question_signals(
        "表格形式呢",
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        "表格形式呢",
        signals=signals,
        state=state,
        current_focus_file=state.last_result_set_focus_file,
        event_name=event.name,
    )

    assert event.name == "answer_presentation_followup"
    assert scope.answer_entity_followup is False
    assert scope.requires_result_set_generation is False
    assert scope.result_scope_paths is None


def test_entity_attribute_then_presentation_then_new_attribute_restores_semantic_authority():
    state = _answer_entity_state("合成项甲与合成项乙已有条件说明。")
    state.last_content_user_question = "有哪些条件？"
    state.last_effective_search_query = "合成项 条件"

    presented = update_state_after_answer_presentation(
        state,
        "给我个表格吧",
        "| 对象 | 条件 |\n| --- | --- |\n| 合成项甲 | 已说明 |",
    )
    question = "都有时间限制吗？"
    event = detect_dialog_event(question, presented, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=presented.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=presented,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name == "content_followup"
    assert scope.answer_entity_followup is True
    assert scope.query_result_set_items == ("合成项甲", "合成项乙")
    assert scope.result_scope_paths == ("资料甲.md", "资料乙.md")
