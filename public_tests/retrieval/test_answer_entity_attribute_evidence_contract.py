from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.retrieval_flow.materials import (
    build_retrieval_materials,
    build_safe_final_prompt,
)
from retrieval.attribute_evidence import extract_requested_attribute_terms
from retrieval.search_intent import determine_query_flags


class _CaptureLogger:
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, message: str):
        self.messages.append(message)

    def info(self, message: str):
        self.messages.append(message)

    def warning(self, message: str):
        self.messages.append(message)


class _EmbeddingStub:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=float)


def _dense_single_file_state(
    *,
    path: str,
    entities: tuple[str, str],
    evidence_lines: tuple[str, str],
):
    chunk_texts = [
        f"{entities[0]}的总体介绍。",
        "第一段通用背景。",
        f"{entities[1]}的总体介绍。",
        "第二段通用背景。",
        "第三段通用背景。",
        evidence_lines[0],
        "附录中的通用说明。",
        evidence_lines[1],
    ]
    return SimpleNamespace(
        paths=[path],
        docs=["\n".join(chunk_texts)],
        chunk_paths=[path] * len(chunk_texts),
        chunk_texts=chunk_texts,
        chunk_file_times=[datetime.now()] * len(chunk_texts),
        chunk_embeddings=np.asarray(
            [[0.90 - index * 0.05, 0.0] for index in range(len(chunk_texts))],
            dtype=float,
        ),
        chunk_meta=[
            {
                "chunk_id": index,
                "start": index * 100,
                "end": index * 100 + len(text),
            }
            for index, text in enumerate(chunk_texts)
        ],
    )


@pytest.mark.parametrize(
    ("path", "entities", "question", "evidence_lines", "attribute_term"),
    [
        (
            "招聘资料.md",
            ("合成岗位甲", "合成岗位乙"),
            "35岁以上呢？",
            ("合成岗位甲的年龄要求为42岁。", "合成岗位乙接受38岁申请者。"),
            "岁",
        ),
        (
            "合同资料.md",
            ("合成交付事项", "合成验收事项"),
            "有时间限制吗？",
            ("合成交付事项的时间限制为三十日。", "合成验收事项的时间限制为十日。"),
            "时间限制",
        ),
        (
            "采购资料.md",
            ("合成设备甲", "合成设备乙"),
            "都有保修期吗？",
            ("合成设备甲的保修期为两年。", "合成设备乙的保修期为一年。"),
            "保修期",
        ),
        (
            "课程资料.md",
            ("合成课程甲", "合成课程乙"),
            "有没有先修条件？",
            ("合成课程甲的先修条件是基础课程。", "合成课程乙的先修条件是入门课程。"),
            "先修条件",
        ),
    ],
)
def test_answer_entity_attribute_followup_prioritizes_late_joint_evidence(
    path,
    entities,
    question,
    evidence_lines,
    attribute_term,
):
    repo_state = _dense_single_file_state(
        path=path,
        entities=entities,
        evidence_lines=evidence_lines,
    )
    logger = _CaptureLogger()

    materials = build_retrieval_materials(
        question=question,
        search_query=" ".join([*entities, question]),
        context_anchor="",
        flags=determine_query_flags(question),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None,
        event=SimpleNamespace(name="content_followup"),
        allowed_paths={path},
        answer_entity_followup=True,
        answer_entity_items=list(entities),
    )

    assert attribute_term in extract_requested_attribute_terms(question)
    assert materials["relevant_indices"][:2] == [5, 7]
    assert all(line in materials["context_text"] for line in evidence_lines)
    assert any("实体属性联合证据" in message for message in logger.messages)
    assert any("总片段限制为 14" in message for message in logger.messages)


def test_plain_retrieval_does_not_enable_entity_attribute_secondary_search():
    path = "通用资料.md"
    repo_state = _dense_single_file_state(
        path=path,
        entities=("合成对象甲", "合成对象乙"),
        evidence_lines=("合成对象甲的条件位于后文。", "合成对象乙的条件位于后文。"),
    )
    logger = _CaptureLogger()

    materials = build_retrieval_materials(
        question="条件是什么？",
        search_query="条件",
        context_anchor="",
        flags=determine_query_flags("条件是什么？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None,
        event=SimpleNamespace(name="normal_retrieval"),
    )

    assert not any("实体属性联合证据" in message for message in logger.messages)
    assert not any("实体属性上下文预算" in message for message in logger.messages)
    assert materials["relevant_indices"][0] == 0


def test_attribute_evidence_keeps_an_anchor_for_each_unmatched_allowed_file():
    now = datetime.now()
    paths = ["事项资料.md", "背景资料.md"]
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["", ""],
        chunk_paths=[paths[0], paths[0], paths[1]],
        chunk_texts=[
            "合成事项甲的总体介绍。",
            "合成事项甲的时间限制为三十日。",
            "这里只记录另一来源的背景。",
        ],
        chunk_file_times=[now, now, now],
        chunk_embeddings=np.asarray(
            [[0.90, 0.0], [0.45, 0.0], [0.10, 0.0]],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 10},
            {"chunk_id": 1, "start": 10, "end": 20},
            {"chunk_id": 0, "start": 0, "end": 10},
        ],
    )

    materials = build_retrieval_materials(
        question="有时间限制吗？",
        search_query="合成事项甲 有时间限制吗",
        context_anchor="",
        flags=determine_query_flags("有时间限制吗？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=_CaptureLogger(),
        current_focus_file=None,
        event=SimpleNamespace(name="content_followup"),
        allowed_paths=set(paths),
        answer_entity_followup=True,
        answer_entity_items=["合成事项甲"],
    )

    assert materials["relevant_indices"][:2] == [1, 2]
    assert all(f"文件【{path}】" in materials["context_text"] for path in paths)


def test_absence_claim_contract_distinguishes_source_negative_from_retrieval_gap():
    prompt = build_safe_final_prompt(
        memory_buffer=[],
        current_focus_file=None,
        inventory_candidates_text="",
        context_text="【参考片段】: 当前片段只覆盖部分章节。",
        timeline_evidence_text="",
        question="这些事项都有时间限制吗？",
        event_name="content_followup",
        result_set_items=["合成事项甲", "合成事项乙"],
        result_set_entity_type="事项",
        answer_entity_followup=True,
    )

    assert "明确给出否定、排除或完整适用范围" in prompt
    assert "有限检索没有命中不等于原始资料没有" in prompt
    assert "当前检索到的证据中暂未找到明确说明" in prompt
    assert "不得把检索缺口写成来源事实" in prompt
    assert "都不能自动推出其他群体被排除" in prompt
    assert "仅限、不得、不适用、不包括" in prompt
