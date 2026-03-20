from __future__ import annotations

import re

import numpy as np

from ai.capability_common import extract_fine_topics



def extract_tag_buckets(repo_state):
    return extract_fine_topics(repo_state)



def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)



def find_files_by_semantic_tag(repo_state, model_emb, query: str, top_k_tags: int = 5, limit: int = 50):
    if model_emb is None:
        return [], []

    buckets = extract_tag_buckets(repo_state)
    if not buckets:
        return [], []

    tag_texts = [item["tag"] for item in buckets]
    query_vec = model_emb.encode([query])[0]
    tag_vecs = model_emb.encode(tag_texts)

    scored = []
    for item, vec in zip(buckets, tag_vecs):
        sim = cosine_sim(query_vec, vec)
        scored.append((item, sim))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    selected = []
    for idx, (item, sim) in enumerate(scored[:top_k_tags]):
        if idx == 0 or sim >= 0.35:
            selected.append((item, sim))
    if not selected:
        return [], []

    merged_paths: list[str] = []
    seen: set[str] = set()
    for item, _sim in selected:
        for path in item["paths"]:
            if path not in seen:
                seen.add(path)
                merged_paths.append(path)

    matched_tags = [item["tag"] for item, _sim in selected]
    return merged_paths[:limit], matched_tags



def build_tag_clusters(repo_state, model_emb, sim_threshold: float = 0.65):
    buckets = extract_tag_buckets(repo_state)
    if not buckets:
        return []

    tags = [item["tag"] for item in buckets]
    vecs = model_emb.encode(tags)

    clusters = []
    used = set()

    for i, (tag, vec) in enumerate(zip(tags, vecs)):
        if i in used:
            continue

        cluster = {"tags": [tag], "paths": set(buckets[i]["paths"])}
        used.add(i)

        for j in range(i + 1, len(tags)):
            if j in used:
                continue
            sim = cosine_sim(vec, vecs[j])
            if sim >= sim_threshold:
                cluster["tags"].append(tags[j])
                cluster["paths"].update(buckets[j]["paths"])
                used.add(j)

        clusters.append(cluster)

    clusters.sort(key=lambda cluster: len(cluster["paths"]), reverse=True)
    return clusters



def generate_cluster_label(cluster, topic_summarizer):
    if not topic_summarizer:
        return cluster["tags"][0]

    top_tags = cluster["tags"][:6]
    prompt = (
        "下面是一组相关标签，请给它们起一个简洁的人类可读类别名称。\n"
        "要求：\n"
        "1. 2~6个字\n"
        "2. 必须是一个自然类别\n"
        "3. 不要解释\n\n"
        "标签：\n" + "、".join(top_tags)
    )

    try:
        result = topic_summarizer(prompt).strip()
        if not result:
            return top_tags[0]

        result = result.replace("\n", "").strip("：: ")
        if len(result) > 10:
            return top_tags[0]
        return result
    except Exception:
        return top_tags[0]



def score_file_against_query(query: str, path: str, shadow_tags: str) -> float:
    q = (query or "").strip().lower()
    path_l = (path or "").lower()
    tags_l = (shadow_tags or "").lower()

    if not q:
        return 0.0

    score = 0.0
    if q in path_l:
        score += 6.0
    if q in tags_l:
        score += 5.0

    parts = [part for part in re.split(r"[\s/、，,]+", q) if len(part) >= 2]
    if not parts:
        parts = [q]

    for part in parts:
        if part in path_l:
            score += 3.0
        if part in tags_l:
            score += 2.5

    path_len = len(path_l)
    if path_len <= 20:
        score += 0.8
    elif path_len <= 40:
        score += 0.4

    return score



def rerank_paths_in_cluster(repo_state, query: str, paths: list[str], limit: int = 50) -> list[str]:
    path_to_record = {str(record.get("path", "")): record for record in getattr(repo_state, "doc_records", [])}

    scored = []
    for path in paths:
        record = path_to_record.get(path, {})
        shadow_tags = str(record.get("shadow_tags", "") or "")
        score = score_file_against_query(query, path, shadow_tags)
        scored.append((path, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [path for path, _score in scored[:limit]]



def find_files_by_semantic_cluster(repo_state, model_emb, query: str, topic_summarizer=None, limit: int = 50):
    if model_emb is None:
        return [], [], None

    clusters = build_tag_clusters(repo_state, model_emb)
    if not clusters:
        return [], [], None

    query_vec = model_emb.encode([query])[0]
    scored = []

    for cluster in clusters:
        if len(cluster["paths"]) < 2:
            continue

        tag_vecs = model_emb.encode(cluster["tags"])
        center_vec = np.mean(tag_vecs, axis=0)
        sim = cosine_sim(query_vec, center_vec)
        scored.append((cluster, sim))

    if not scored:
        return [], [], None

    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_cluster, best_sim = scored[0]

    if best_sim < 0.35:
        return [], [], None

    label = generate_cluster_label(best_cluster, topic_summarizer)
    cluster_info = {
        "label": label,
        "tags": best_cluster["tags"],
        "paths": best_cluster["paths"],
    }

    paths = rerank_paths_in_cluster(repo_state, query, list(best_cluster["paths"]), limit=limit)
    return paths, cluster_info["tags"], cluster_info
