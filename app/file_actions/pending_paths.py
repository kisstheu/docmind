from __future__ import annotations


def replace_result_set_paths(items: list[str] | None, rename_map: dict[str, str]) -> list[str] | None:
    if not items:
        return items
    return [rename_map.get(item, item) for item in items]
