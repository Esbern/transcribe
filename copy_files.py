import os
import sys
import yaml
import shutil


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def fix_split_paths(files_list):
    """Fixes recipe entries where a single path was accidentally split across two YAML items."""
    assembled = []
    for s in files_list or []:
        if not isinstance(s, str):
            continue
        cur = s.strip()
        merged = False

        if assembled:
            prev = assembled[-1]
            candidate = prev + cur
            if path_exists(candidate):
                assembled[-1] = candidate
                merged = True
            else:
                candidate2 = prev + cur.lstrip("-").strip()
                if path_exists(candidate2):
                    assembled[-1] = candidate2
                    merged = True
                else:
                    candidate3 = prev.rstrip() + " " + cur.lstrip("-").strip()
                    if path_exists(candidate3):
                        assembled[-1] = candidate3
                        merged = True

        if not merged:
            assembled.append(cur)

    return assembled


def main():
    cfg = read_yaml("./scanner.yml")
    recipe_path = cfg["paths"]["recipe_file"]
    dst_folder = "D:\\lydfiler"

    recipe = read_yaml(recipe_path)
    if not isinstance(recipe, list):
        print("Recipe file must be a list of entries.")
        return 2

    os.makedirs(dst_folder, exist_ok=True)

    total_files = 0
    copied = 0

    for entry in recipe:
        entry_id = str(entry.get("id"))
        raw_files = entry.get("files") or []
        files = fix_split_paths(raw_files)

        for src in files:
            total_files += 1
            if not path_exists(src):
                print(f"[{entry_id}] File not found: {src}")
                continue

            filename = os.path.basename(src)
            dst = os.path.join(dst_folder, filename)

            try:
                shutil.copy2(src, dst)
                print(f"[{entry_id}] Copied: {filename}")
                copied += 1
            except Exception as e:
                print(f"[{entry_id}] Error copying {filename}: {e}")

    print(f"\nCopied {copied}/{total_files} files to {dst_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
