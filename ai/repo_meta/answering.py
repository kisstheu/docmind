from __future__ import annotations

from ai.repo_meta.answering_parts.naming import _answer_name_content_mismatch
from ai.repo_meta.answering_parts.size import (
    _answer_count,
    _answer_format,
    _answer_size_consistency,
    _answer_total_size,
    calc_repo_total_bytes,
)
from ai.repo_meta.answering_parts.time import (
    _answer_list_files,
    _answer_list_files_with_time,
    _answer_time,
)
from ai.repo_meta.category import (
    answer_repo_content_category_count_breakdown_question,
    answer_repo_content_category_confirm_question,
    answer_repo_content_category_label_list_question,
    answer_repo_content_category_overview_question,
    answer_repo_content_category_question,
    answer_repo_content_category_summary_question,
)
from ai.repo_meta.classifier import classify_repo_meta_question, extract_topic_from_list_request
from ai.repo_meta.semantic import find_files_by_semantic_cluster


def _answer_list_files_by_topic(
    question: str,
    repo_state,
    model_emb=None,
    topic_summarizer=None,
    category_context_answer: str | None = None,
) -> tuple[str, str]:
    target_topic = extract_topic_from_list_request(question)
    category_answer = answer_repo_content_category_label_list_question(
        target_topic=target_topic,
        repo_state=repo_state,
        previous_summary=category_context_answer,
        topic_summarizer=topic_summarizer,
    )
    if category_answer:
        return category_answer, "list_files_by_topic"

    matched, _matched_tags, best_cluster = find_files_by_semantic_cluster(
        repo_state,
        model_emb,
        target_topic,
        topic_summarizer=topic_summarizer,
        limit=50,
    )

    if not matched:
        return f"当前知识库里暂时没有明显命中\"{target_topic}\"的文件。", "list_files_by_topic"

    label = best_cluster["label"] if best_cluster else "相关内容"
    lines = [f"按语义上最接近的类别（{label}）来看，相关文件大约有 {len(matched)} 个，先列出这些："]
    lines.extend(f"- {path}" for path in matched)
    return "\n".join(lines), "list_files_by_topic"


def answer_repo_meta_question(
    question: str,
    repo_state,
    model_emb=None,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
    last_local_answer: str | None = None,
    category_context_answer: str | None = None,
    topic_summarizer=None,
):
    paths = list(repo_state.paths)
    all_files = list(repo_state.all_files)
    file_times = list(repo_state.file_times)

    if not paths:
        return "当前知识库里还没有可用文档。", "empty"

    topic = classify_repo_meta_question(
        question,
        last_user_question=last_user_question,
        last_local_topic=last_local_topic,
    )

    if topic == "count":
        return _answer_count(paths)
    if topic == "total_size":
        return _answer_total_size(repo_state)
    if topic == "size_consistency":
        return _answer_size_consistency(question, repo_state, last_user_question=last_user_question)
    if topic == "format":
        return _answer_format(all_files)
    if topic == "list_files_by_topic":
        return _answer_list_files_by_topic(
            question,
            repo_state,
            model_emb=model_emb,
            topic_summarizer=topic_summarizer,
            category_context_answer=category_context_answer,
        )
    if topic == "time":
        return _answer_time(question, paths, file_times)
    if topic == "name_content_mismatch":
        return _answer_name_content_mismatch(repo_state)
    if topic == "list_files":
        return _answer_list_files(paths)
    if topic == "list_files_with_time":
        return _answer_list_files_with_time(paths, file_times)
    if topic == "category":
        return answer_repo_content_category_question(repo_state), topic
    if topic == "category_summary":
        return answer_repo_content_category_summary_question(repo_state, topic_summarizer=topic_summarizer), topic
    if topic == "category_count_breakdown":
        return answer_repo_content_category_count_breakdown_question(
            repo_state,
            previous_summary=last_local_answer,
            topic_summarizer=topic_summarizer,
        ), topic
    if topic == "category_overview":
        return answer_repo_content_category_overview_question(
            repo_state,
            topic_summarizer=topic_summarizer,
            previous_summary=last_local_answer,
        ), topic
    if topic == "category_confirm":
        return answer_repo_content_category_confirm_question(question, repo_state), topic

    return "我识别到你在问知识库的文件信息，但暂时没分清是数量、格式、时间还是列表。你可以换个更直接的问法。", "unknown_repo_meta"
