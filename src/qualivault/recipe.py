# -*- coding: utf-8 -*-

from __future__ import annotations

import yaml
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def generate_recipe(interviews: Dict[str, List[str]], base_path: "Path | str | None" = None) -> List[Dict[str, Any]]:
    """Generate a clean recipe from scan results.

    `interviews` is mapping: id -> list[filepaths]
    If `base_path` is given, file paths are stored relative to it (e.g. `org_audio_folder`)
    so the recipe stays portable across machines.
    """
    base = Path(base_path).expanduser().resolve() if base_path else None
    recipe: List[Dict[str, Any]] = []
    for i_id, files in interviews.items():
        files_sorted = sorted(files)
        if base is not None:
            files_sorted = [str(Path(f).resolve().relative_to(base)) for f in files_sorted]
        recipe.append(
            {
                "id": str(i_id),
                # Fill in per interview (number of people talking); used for diarization.
                "speakers": None,
                "files": files_sorted,
                "output_name": f"Interview_{i_id}.flac",
            }
        )
    return recipe


def check_recipe(
    recipe: List[Dict[str, Any]],
    org_audio_folder: "Path | str | None" = None,
) -> Tuple[List[str], List[str]]:
    """Sanity-check a recipe before processing. Returns (errors, warnings).

    Errors are problems that make the pipeline silently skip or mix up interviews:
    duplicate ids, two items writing the same output file, missing/empty source files,
    or one id collecting files from several folders (two folders with the same id).
    """
    errors: List[str] = []
    warnings: List[str] = []
    base = Path(org_audio_folder).expanduser() if org_audio_folder else None

    ids = Counter(str(it.get("id")) for it in recipe)
    for iid, n in ids.items():
        if n > 1:
            errors.append(f"id {iid!r} appears {n} times")

    outputs = Counter(str(it.get("output_name")) for it in recipe)
    for name, n in outputs.items():
        if n > 1:
            errors.append(f"output_name {name!r} is used by {n} items (all but the first would be skipped)")

    for it in recipe:
        label = f"id {it.get('id')!r}"
        if not it.get("output_name"):
            errors.append(f"{label}: missing output_name")
        files = it.get("files") or []
        if not files:
            errors.append(f"{label}: no audio files")
            continue
        folders = {str(Path(f).parent) for f in files}
        if len(folders) > 1:
            errors.append(f"{label}: files come from {len(folders)} different folders {sorted(folders)}")
        if base is not None:
            for f in files:
                path = Path(f) if Path(f).is_absolute() else base / f
                if not path.exists():
                    errors.append(f"{label}: file not found: {path}")
        if len(files) > 1:
            warnings.append(f"{label}: {len(files)} files will be joined in name order: {[Path(f).name for f in files]}")
        if it.get("speakers") in (None, ""):
            warnings.append(f"{label}: 'speakers' not set")

    return errors, warnings


def assert_recipe_ok(recipe: List[Dict[str, Any]], org_audio_folder: "Path | str | None" = None) -> None:
    """Print warnings and raise ValueError listing every error found by `check_recipe`."""
    errors, warnings = check_recipe(recipe, org_audio_folder)
    for w in warnings:
        print(f"⚠️  {w}")
    if errors:
        raise ValueError("Recipe problems (fix processing_recipe.yaml first):\n  - " + "\n  - ".join(errors))
