import os
import sys
import yaml
import shutil
import tempfile
import subprocess
from datetime import datetime


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def win_long_path(p: str) -> str:
    # Add \\?\ prefix for long Windows paths
    if os.name == "nt":
        p = os.path.abspath(p)
        if not p.startswith("\\\\?\\"):
            # Only prefix for absolute drive-letter paths or UNC
            if (
                (len(p) > 2 and p[1] == ":" and p[2] == "\\")
                or p.startswith("\\\\")
            ):
                return "\\\\?\\" + p
    return p


def path_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def fix_split_paths(files_list):
    """Fixes recipe entries where a single path was accidentally split across two YAML items.
    Heuristic: if current item + previous assembled item forms an existing path, merge them.
    """
    assembled = []
    for s in files_list or []:
        if not isinstance(s, str):
            continue
        cur = s.strip()
        merged = False

        if assembled:
            prev = assembled[-1]
            # Try direct concatenation (common case: prev ends with 'files' and cur starts with ' - Lydfiler\\')
            candidate = prev + cur
            if path_exists(candidate):
                assembled[-1] = candidate
                merged = True
            else:
                # Try concatenation removing a leading '-' and extra spaces
                candidate2 = prev + cur.lstrip("-").strip()
                if path_exists(candidate2):
                    assembled[-1] = candidate2
                    merged = True
                else:
                    # Try adding a separating space
                    candidate3 = prev.rstrip() + " " + cur.lstrip("-").strip()
                    if path_exists(candidate3):
                        assembled[-1] = candidate3
                        merged = True

        if not merged:
            assembled.append(cur)

    return assembled


def run_ffmpeg(args):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + args
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {' '.join(cmd)}\n  -> {e}")
        return False


def convert_to_flac(src_path, dst_path, force_rate=48000, force_channels=2):
    src = win_long_path(src_path)
    dst = win_long_path(dst_path)
    args = ["-y", "-i", src, "-ar", str(force_rate), "-ac", str(force_channels), dst]
    return run_ffmpeg(args)


def concat_flacs(file_list, output_path):
    # Create a concat list file
    with tempfile.TemporaryDirectory() as td:
        list_path = os.path.join(td, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as lf:
            for p in file_list:
                # ffmpeg concat demuxer expects lines like: file 'path'
                lf.write(f"file '{win_long_path(os.path.abspath(p))}'\n")

        out = win_long_path(output_path)
        # Use concat demuxer; input files are homogeneous FLACs so copy is safe
        args = ["-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "flac", out]
        return run_ffmpeg(args)


def process_entry(entry, output_base):
    entry_id = str(entry.get("id"))
    output_name = entry.get("output_name") or f"interview_{entry_id}.flac"
    output_path = os.path.join(output_base, output_name)

    raw_files = entry.get("files") or []
    files = fix_split_paths(raw_files)

    # Keep only existing files and preserve order
    existing = [f for f in files if path_exists(f)]
    if not existing:
        print(f"[{entry_id}] No valid input files found; skipping.")
        entry["status"] = "missing"
        return False

    ensure_dir(output_base)

    # If only one file, convert directly
    if len(existing) == 1:
        print(f"[{entry_id}] Converting 1 file -> {output_name}")
        ok = convert_to_flac(existing[0], output_path)
        entry["status"] = "done" if ok else "error"
        entry["processed_at"] = datetime.now().isoformat(timespec="seconds")
        return ok

    # Multiple files: normalize to temp FLACs then concat
    print(f"[{entry_id}] Concatenating {len(existing)} files -> {output_name}")
    tmp_dir = tempfile.mkdtemp(prefix=f"recipe_{entry_id}_")
    tmp_flacs = []
    try:
        for idx, src in enumerate(existing, start=1):
            tmp_flac = os.path.join(tmp_dir, f"part_{idx:03d}.flac")
            ok = convert_to_flac(src, tmp_flac)
            if not ok:
                entry["status"] = "error"
                entry["processed_at"] = datetime.now().isoformat(timespec="seconds")
                return False
            tmp_flacs.append(tmp_flac)

        ok = concat_flacs(tmp_flacs, output_path)
        entry["status"] = "done" if ok else "error"
        entry["processed_at"] = datetime.now().isoformat(timespec="seconds")
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    # Load config
    cfg = read_yaml("./scanner.yml")
    recipe_path = cfg["paths"]["recipe_file"]
    output_base = cfg["paths"]["output_base_folder"]

    recipe = read_yaml(recipe_path)
    if not isinstance(recipe, list):
        print("Recipe file must be a list of entries.")
        return 2

    total = len(recipe)
    done = 0
    for entry in recipe:
        status = (entry.get("status") or "").lower()
        # Process all entries; skip only if explicitly 'done' and file exists
        if status == "done":
            done += 1
            continue

        ok = process_entry(entry, output_base)
        if ok:
            done += 1

        # Save progress incrementally to be resilient
        write_yaml(recipe_path, recipe)

    print(f"Completed {done}/{total} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
