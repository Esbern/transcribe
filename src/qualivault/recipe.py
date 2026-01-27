# -*- coding: utf-8 -*-

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Iterable, List


RUNTIME_STATUS_KEYS = {
    "status",
    "last_good_status",
    "convert_status",
    "analysis_status",
    "transcribe_status",
    "error_msg",
    "error_message",
    "transcript_path",
}


def load_recipe(recipe_path: Path) -> List[Dict[str, Any]]:
    recipe_path = Path(recipe_path)
    if not recipe_path.exists():
        return []
    with open(recipe_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"Recipe must be a list, got: {type(data).__name__}")
    return data


def save_recipe(recipe_path: Path, recipe: List[Dict[str, Any]]) -> None:
    recipe_path = Path(recipe_path)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recipe_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f, sort_keys=False, allow_unicode=True)


def interview_key(item: Dict[str, Any]) -> str:
    """Stable interview key used across steps."""
    output_name = item.get("output_name")
    if output_name:
        return Path(str(output_name)).stem
    if "id" in item:
        return f"Interview_{item['id']}"
    raise KeyError("Recipe item missing output_name/id")


def expected_flac_name(item: Dict[str, Any]) -> str:
    output_name = item.get("output_name")
    if not output_name:
        raise KeyError("Recipe item missing output_name")
    return str(output_name)


def expected_csv_name(item: Dict[str, Any]) -> str:
    return expected_flac_name(item).replace(".flac", ".csv")


def strip_runtime_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(item)
    for k in list(cleaned.keys()):
        if k in RUNTIME_STATUS_KEYS:
            cleaned.pop(k, None)
    return cleaned


def generate_recipe(interviews: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Generate a clean recipe from scan results.

    `interviews` is mapping: id -> list[filepaths]
    """
    recipe: List[Dict[str, Any]] = []
    for i_id, files in interviews.items():
        files_sorted = sorted(files)
        recipe.append(
            {
                "id": str(i_id),
                "files": files_sorted,
                "output_name": f"Interview_{i_id}.flac",
            }
        )
    return recipe
