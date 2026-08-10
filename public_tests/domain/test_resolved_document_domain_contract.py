from types import SimpleNamespace

import pytest

from app.resolved_document_domain import (
    ResolvedRepoDocument,
    has_unique_repo_display_name,
    resolve_repo_document,
)


def test_exact_unique_path_maps_to_aligned_full_repo_document():
    repo_state = SimpleNamespace(
        paths=["资料/合成甲.md", "资料/合成乙.md"],
        docs=["完整正文甲", "完整正文乙尾部约束"],
    )

    assert resolve_repo_document(repo_state, "资料/合成乙.md") == ResolvedRepoDocument(
        path="资料/合成乙.md",
        text="完整正文乙尾部约束",
    )


@pytest.mark.parametrize(
    "repo_state,resolved_path",
    [
        (SimpleNamespace(paths=["资料/合成甲.md"], docs=[]), "资料/合成甲.md"),
        (
            SimpleNamespace(
                paths=["资料/合成甲.md", "资料/合成甲.md"],
                docs=["正文甲", "正文乙"],
            ),
            "资料/合成甲.md",
        ),
        (SimpleNamespace(paths=["资料/合成甲.md"], docs=[""]), "资料/合成甲.md"),
        (SimpleNamespace(paths=["资料/合成甲.md"], docs=["正文甲"]), "资料/缺失.md"),
    ],
)
def test_full_document_mapping_fails_closed_when_alignment_is_not_safe(
    repo_state,
    resolved_path,
):
    assert resolve_repo_document(repo_state, resolved_path) is None


def test_retrieval_inferred_focus_requires_unique_display_stem():
    repo_state = SimpleNamespace(
        paths=["目录甲/同名资料.md", "目录乙/同名资料.md", "目录丙/唯一资料.md"],
    )

    assert has_unique_repo_display_name(repo_state, "目录甲/同名资料.md") is False
    assert has_unique_repo_display_name(repo_state, "目录丙/唯一资料.md") is True
