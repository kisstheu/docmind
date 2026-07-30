from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from retrieval.search_context import build_context_text


_CONTEXT_HEADER_RE = re.compile(r"文件【(?P<path>.+?)】（chunk #(?P<chunk_id>\d+)，")


def _build_repo_state(file_chunk_counts: list[tuple[str, int]]):
    chunk_paths: list[str] = []
    chunk_texts: list[str] = []
    chunk_meta: list[dict[str, int | str]] = []

    for path, chunk_count in file_chunk_counts:
        for chunk_id in range(chunk_count):
            text = f"{path} 的合成段落 {chunk_id}。"
            chunk_paths.append(path)
            chunk_texts.append(text)
            chunk_meta.append(
                {
                    "path": path,
                    "chunk_id": chunk_id,
                    "start": chunk_id * 100,
                    "end": chunk_id * 100 + len(text),
                }
            )

    return SimpleNamespace(
        chunk_paths=chunk_paths,
        chunk_texts=chunk_texts,
        chunk_meta=chunk_meta,
    )


def _global_index(repo_state, path: str, chunk_id: int) -> int:
    for idx, (candidate_path, meta) in enumerate(
        zip(repo_state.chunk_paths, repo_state.chunk_meta)
    ):
        if candidate_path == path and meta["chunk_id"] == chunk_id:
            return idx
    raise AssertionError(f"missing synthetic chunk: {path} #{chunk_id}")


def _selected_chunks(context_text: str) -> list[tuple[str, int]]:
    return [
        (match.group("path"), int(match.group("chunk_id")))
        for match in _CONTEXT_HEADER_RE.finditer(context_text)
    ]


@pytest.mark.parametrize(
    "path",
    [
        "设备安装手册.md",
        "通用合规条例.txt",
        "车辆维护资料.pdf",
    ],
)
def test_late_rank_one_seed_survives_earlier_low_priority_region(path: str) -> None:
    repo_state = _build_repo_state([(path, 22)])
    relevant_indices = [
        _global_index(repo_state, path, 20),
        _global_index(repo_state, path, 2),
    ]

    context_text = build_context_text(
        relevant_indices,
        repo_state,
        logging.getLogger(__name__),
    )
    selected_ids = [
        chunk_id
        for selected_path, chunk_id in _selected_chunks(context_text)
        if selected_path == path
    ]

    assert 20 in selected_ids
    assert 2 in selected_ids
    assert any(chunk_id in {19, 21} for chunk_id in selected_ids)
    assert selected_ids == sorted(selected_ids)


def test_multiple_seeds_are_kept_before_any_neighbor_uses_budget() -> None:
    path = "系统运维指南.md"
    repo_state = _build_repo_state([(path, 52)])
    relevant_indices = [
        _global_index(repo_state, path, 20),
        _global_index(repo_state, path, 50),
    ]

    context_text = build_context_text(
        relevant_indices,
        repo_state,
        logging.getLogger(__name__),
    )
    selected_ids = [chunk_id for _, chunk_id in _selected_chunks(context_text)]

    assert len(selected_ids) == 3
    assert {20, 50}.issubset(selected_ids)
    assert len(set(selected_ids) - {20, 50}) == 1
    assert selected_ids == sorted(selected_ids)


def test_each_file_keeps_its_late_seed_under_per_file_budget() -> None:
    first_path = "规范甲.txt"
    second_path = "规范乙.txt"
    repo_state = _build_repo_state(
        [
            (first_path, 31),
            (second_path, 31),
        ]
    )
    relevant_indices = [
        _global_index(repo_state, first_path, 20),
        _global_index(repo_state, second_path, 25),
        _global_index(repo_state, first_path, 2),
        _global_index(repo_state, second_path, 3),
    ]

    context_text = build_context_text(
        relevant_indices,
        repo_state,
        logging.getLogger(__name__),
    )
    selected = _selected_chunks(context_text)
    first_ids = [chunk_id for path, chunk_id in selected if path == first_path]
    second_ids = [chunk_id for path, chunk_id in selected if path == second_path]

    assert len(first_ids) == 3
    assert len(second_ids) == 3
    assert {2, 20}.issubset(first_ids)
    assert {3, 25}.issubset(second_ids)
    assert first_ids == sorted(first_ids)
    assert second_ids == sorted(second_ids)


def test_neighbors_cannot_displace_any_seed_when_seeds_fill_budget() -> None:
    path = "通用长文档.md"
    repo_state = _build_repo_state([(path, 42)])
    relevant_indices = [
        _global_index(repo_state, path, 40),
        _global_index(repo_state, path, 10),
        _global_index(repo_state, path, 25),
    ]

    context_text = build_context_text(
        relevant_indices,
        repo_state,
        logging.getLogger(__name__),
    )
    selected_ids = [chunk_id for _, chunk_id in _selected_chunks(context_text)]

    assert selected_ids == [10, 25, 40]


@pytest.mark.parametrize(
    ("chunk_count", "seed_chunk_id", "expected_ids"),
    [
        (1, 0, [0]),
        (3, 1, [0, 1, 2]),
    ],
)
def test_simple_context_within_budget_keeps_existing_local_order(
    chunk_count: int,
    seed_chunk_id: int,
    expected_ids: list[int],
) -> None:
    path = "短资料.md"
    repo_state = _build_repo_state([(path, chunk_count)])

    context_text = build_context_text(
        [_global_index(repo_state, path, seed_chunk_id)],
        repo_state,
        logging.getLogger(__name__),
    )

    assert [chunk_id for _, chunk_id in _selected_chunks(context_text)] == expected_ids
