from __future__ import annotations

from ai.capability_common import format_bytes
from ai.repo_meta.category import (
    answer_repo_content_category_confirm_question,
    answer_repo_content_category_question,
    answer_repo_content_category_summary_question,
)
from ai.repo_meta.classifier import classify_repo_meta_question, extract_topic_from_list_request
from ai.repo_meta.semantic import find_files_by_semantic_cluster



def calc_repo_total_bytes(repo_state) -> int:
    doc_records = getattr(repo_state, "doc_records", None) or []
    if not doc_records:
        return 0

    total_bytes = 0
    for record in doc_records:
        if not isinstance(record, dict):
            continue
        total_bytes += int(record.get("file_size", 0) or record.get("size", 0) or record.get("bytes", 0) or 0)

    return total_bytes



def _answer_count(paths: list[str]) -> tuple[str, str]:
    return f"当前知识库共有 {len(paths)} 个文件。", "count"



def _answer_total_size(repo_state) -> tuple[str, str]:
    total_bytes = calc_repo_total_bytes(repo_state)
    return f"当前知识库里这些文档总共约占 {format_bytes(total_bytes)} 空间。", "total_size"



def _answer_format(all_files) -> tuple[str, str]:
    suffixes = sorted({file.suffix.lower() or "[无后缀]" for file in all_files})
    answer = "当前知识库中的文件格式有：\n" + "\n".join(f"- {suffix}" for suffix in suffixes)
    return answer, "format"



def _answer_time(paths, file_times) -> tuple[str, str]:
    latest_idx = max(range(len(file_times)), key=lambda i: file_times[i])
    earliest_idx = min(range(len(file_times)), key=lambda i: file_times[i])
    answer = (
        f"最早的文件：{paths[earliest_idx]}（{file_times[earliest_idx].strftime('%Y-%m-%d %H:%M:%S')}）\n"
        f"最新的文件：{paths[latest_idx]}（{file_times[latest_idx].strftime('%Y-%m-%d %H:%M:%S')}）"
    )
    return answer, "time"



def _answer_list_files(paths: list[str]) -> tuple[str, str]:
    show_n = min(50, len(paths))
    preview = "\n".join(f"- {path}" for path in paths[:show_n])
    if len(paths) > show_n:
        answer = (
            f"当前知识库里共有 {len(paths)} 个文件，先列出前 {show_n} 个：\n"
            f"{preview}\n- ...(其余 {len(paths) - show_n} 个未展开)"
        )
    else:
        answer = f"当前知识库里的文件如下：\n{preview}"
    return answer, "list_files"



def _answer_list_files_by_topic(question: str, repo_state, model_emb=None, topic_summarizer=None) -> tuple[str, str]:
    target_topic = extract_topic_from_list_request(question)
    matched, _matched_tags, best_cluster = find_files_by_semantic_cluster(
        repo_state,
        model_emb,
        target_topic,
        topic_summarizer=topic_summarizer,
        limit=50,
    )

    if not matched:
        return f"当前知识库里暂时没有明显命中“{target_topic}”的文件。", "list_files_by_topic"

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
    if topic == "format":
        return _answer_format(all_files)
    if topic == "list_files_by_topic":
        return _answer_list_files_by_topic(
            question,
            repo_state,
            model_emb=model_emb,
            topic_summarizer=topic_summarizer,
        )
    if topic == "time":
        return _answer_time(paths, file_times)
    if topic == "list_files":
        return _answer_list_files(paths)
    if topic == "category":
        return answer_repo_content_category_question(repo_state), topic
    if topic == "category_summary":
        return answer_repo_content_category_summary_question(
            repo_state,
            topic_summarizer=topic_summarizer,
        ), topic
    if topic == "category_confirm":
        return answer_repo_content_category_confirm_question(question, repo_state), topic

    return "我识别到你在问知识库的文件信息，但暂时没分清是数量、格式、时间还是列表。你可以换个更直接的问法。", "unknown_repo_meta"