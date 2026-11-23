#!/usr/bin/env python3
"""Researcher tool: one trimmed branch per interview.

Each interview branch contains ONLY:
  - 01_Raw_Transcripts/<interview>.md      (the original, for reference)
  - 02_Coding_In_Progress/<interview>.md   (the copy the student works on)
  - the recording(s) that transcript links to (Git LFS); all other audio is removed
  - everything else in the vault (tags/codebook, .obsidian, templates, ...)
All other transcripts are removed from that branch, so the student sees one interview.

!!! NEVER merge an interview branch into main with GitLab's Merge button: the branch
!!! records the removal of all other interviews, so merging would delete them from main.
!!! Use `collect` instead, which copies only the finished working file back.

Commands (run from anywhere; --repo points at your local clone of the vault repo):
  setup   [--only Hakan_07 ...] [--dry-run]  create + push one branch per interview
  review  Hakan_07                            list what the student changed + compare link
  collect Hakan_07 [--force]                  put the finished file in 03_Complete on a
                                              collect-<interview> branch -> open MR to main
  sync-template --template <url> [--ref main] copy the shared vault-template files into the
                                              project on a sync-template branch -> open MR

Your own working copy is never touched: all work happens in temporary git worktrees.
"""

from __future__ import annotations

import argparse
import os
import re
from urllib.parse import unquote
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

# --- Adjust to the vault's folder names ---------------------------------------------
RAW = "01_Raw_Transcripts"
WIP = "02_Coding_In_Progress"
DONE = "03_Complete"
MAIN = "main"
AUDIO_EXT = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".webm", ".mp4")
BRANCH_PREFIX = "interview-"   # branch students switch to, e.g. interview-Hakan_07
SETUP_TAG_PREFIX = "setup-"    # tag marking the untouched starting point of a branch
COLLECT_PREFIX = "collect-"    # branch used to bring finished work back to main
SYNC_BRANCH = "sync-template"  # branch used to bring vault-template updates into a project
# Paths owned by the vault template: replaced wholesale on every sync.
TEMPLATE_MANAGED = [".obsidian", ".gitlab", "Templates", ".tools",
                    ".gitignore", ".gitattributes", "Student Guide.md", "Git Introduction.md"]
# Copied only if the project has nothing there yet (each project keeps its own codebook).
TEMPLATE_IF_MISSING = {"Tags/README.md": "Tags/"}
# -------------------------------------------------------------------------------------


def git(repo: Path, *args: str, binary: bool = False, stdin: str | None = None):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin.encode() if stdin is not None else None,
        capture_output=True,
        # Never download Git LFS audio while building branches: the script only
        # deletes audio paths, it never needs their content.
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.decode(errors='replace')}")
    return result.stdout if binary else result.stdout.decode()


def list_files(repo: Path, ref: str, folder: str) -> list[str]:
    out = git(repo, "ls-tree", "-r", "--name-only", "-z", ref, "--", folder + "/")
    return [p for p in out.split("\0") if p]


def remote_branches(repo: Path) -> set[str]:
    out = git(repo, "ls-remote", "--heads", "origin")
    return {line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line}


def web_url(repo: Path) -> str | None:
    """Turn the origin URL into a browsable https URL (credentials stripped)."""
    url = git(repo, "remote", "get-url", "origin").strip()
    m = re.match(r"^git@([^:]+):(.+?)(\.git)?$", url)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = re.match(r"^https?://(?:[^@/]+@)?(.+?)(\.git)?$", url)
    return f"https://{m.group(1)}" if m else None


class Worktree:
    """Temporary checkout so the researcher's own clone is never modified."""

    def __init__(self, repo: Path, branch: str, start: str):
        self.repo, self.branch, self.start = repo, branch, start

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "wt"
        git(self.repo, "worktree", "add", "-q", "-B", self.branch, str(self.path), self.start)
        return self.path

    def __exit__(self, *exc):
        git(self.repo, "worktree", "remove", "--force", str(self.path))
        git(self.repo, "branch", "-D", self.branch)  # pushed already; keep local list tidy
        self._tmp.cleanup()


def cmd_setup(repo: Path, only: list[str] | None, dry_run: bool) -> None:
    git(repo, "fetch", "-q", "--tags", "origin")
    base = f"origin/{MAIN}"
    raw_files = [p for p in list_files(repo, base, RAW) if p.endswith(".md")]
    done_names = {PurePosixPath(p).name for p in list_files(repo, base, DONE)}
    existing = remote_branches(repo)

    for raw_path in raw_files:
        name = PurePosixPath(raw_path).name
        stem = PurePosixPath(raw_path).stem
        branch = BRANCH_PREFIX + stem
        if only and stem not in only:
            continue
        if branch in existing:
            print(f"⏭️  {stem}: branch already exists")
            continue
        if name in done_names:
            print(f"⏭️  {stem}: already in {DONE}")
            continue
        if dry_run:
            print(f"🔍 {stem}: would create {branch}")
            continue

        with Worktree(repo, branch, base) as wt:
            others = [
                p
                for folder in (RAW, WIP, DONE)
                for p in list_files(wt, "HEAD", folder)
                if p != raw_path
            ]
            # Keep only the recording(s) this transcript links to; drop all other audio.
            raw_text = (wt / raw_path).read_text(encoding="utf-8", errors="replace")
            raw_text += unquote(raw_text)
            tracked = [p for p in git(wt, "ls-files", "-z").split("\0") if p]
            audio = [p for p in tracked if p.lower().endswith(AUDIO_EXT)]
            keep_audio = [
                p for p in audio
                if re.search(r"(?:^|[/\[(|\s])" + re.escape(PurePosixPath(p).name), raw_text)
            ]
            others += [p for p in audio if p not in keep_audio]
            if audio and not keep_audio:
                print(f"⚠️  {stem}: no recording linked from the transcript – branch has no audio")
            if others:
                git(wt, "rm", "-q", "--pathspec-from-file=-", stdin="\n".join(others))
            dst = wt / WIP / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((wt / raw_path).read_bytes())
            git(wt, "add", "--", f"{WIP}/{name}")
            git(wt, "commit", "-q", "-m", f"Set up coding branch for {stem}")
            git(wt, "tag", "-f", SETUP_TAG_PREFIX + stem)
            git(wt, "push", "-q", "origin", branch, f"refs/tags/{SETUP_TAG_PREFIX}{stem}")
        print(f"✅ {stem}: pushed {branch}")


def changed_files(repo: Path, stem: str) -> list[str]:
    return [
        p
        for p in git(
            repo, "diff", "--name-only", "-z",
            f"{SETUP_TAG_PREFIX}{stem}", f"origin/{BRANCH_PREFIX}{stem}",
        ).split("\0")
        if p
    ]


def find_name(repo: Path, stem: str) -> str:
    for p in list_files(repo, f"{SETUP_TAG_PREFIX}{stem}", WIP):
        if PurePosixPath(p).stem == stem:
            return PurePosixPath(p).name
    sys.exit(f"No {WIP}/{stem}.* found at tag {SETUP_TAG_PREFIX}{stem}")


def cmd_review(repo: Path, stem: str) -> bool:
    """Print changes since setup. Returns True if only the working file changed."""
    git(repo, "fetch", "-q", "--tags", "origin")
    expected = f"{WIP}/{find_name(repo, stem)}"
    changed = changed_files(repo, stem)
    # Obsidian/plugin settings changes stay on the interview branch and are never
    # collected, so they are harmless: report them, but don't block on them.
    settings = [p for p in changed if p.startswith(".obsidian/")]
    unexpected = [p for p in changed if p != expected and p not in settings]

    print(f"Changes in {BRANCH_PREFIX}{stem} since setup:")
    print(git(repo, "diff", "--stat", f"{SETUP_TAG_PREFIX}{stem}", f"origin/{BRANCH_PREFIX}{stem}") or "  (none)")
    if settings:
        print(f"ℹ️  {len(settings)} Obsidian settings file(s) changed – ignored, they stay on this branch.")
    if unexpected:
        print("⚠️  Files changed OUTSIDE the working copy (originals/tags should stay untouched):")
        for p in unexpected:
            print(f"   - {p}")
    url = web_url(repo)
    if url:
        print(f"🔗 Review in GitLab: {url}/-/compare/{SETUP_TAG_PREFIX}{stem}...{BRANCH_PREFIX}{stem}")
    return not unexpected


def cmd_collect(repo: Path, stem: str, force: bool) -> None:
    clean = cmd_review(repo, stem)
    if not clean and not force:
        sys.exit("Stopped: other files were changed. Only the working copy will be collected; "
                 "re-run with --force once you have checked them.")
    name = find_name(repo, stem)
    content = git(repo, "show", f"origin/{BRANCH_PREFIX}{stem}:{WIP}/{name}", binary=True)
    branch = COLLECT_PREFIX + stem

    with Worktree(repo, branch, f"origin/{MAIN}") as wt:
        dst = wt / DONE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        git(wt, "add", "--", f"{DONE}/{name}")
        git(wt, "commit", "-q", "-m", f"Add completed {stem}")
        git(wt, "push", "-q", "-f", "origin", branch)
    url = web_url(repo)
    print(f"✅ Pushed {branch}. Open and merge the merge request into {MAIN}:")
    if url:
        print(f"   {url}/-/merge_requests/new?merge_request[source_branch]={branch}")
    print(f"   Afterwards you may delete {BRANCH_PREFIX}{stem} (the {SETUP_TAG_PREFIX}{stem} tag keeps history).")


def cmd_sync_template(repo: Path, template_url: str, template_ref: str) -> None:
    git(repo, "fetch", "-q", "origin")
    git(repo, "fetch", "-q", template_url, template_ref)
    sha = git(repo, "rev-parse", "FETCH_HEAD").strip()
    template_files = [p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", sha).split("\0") if p]

    with Worktree(repo, SYNC_BRANCH, f"origin/{MAIN}") as wt:
        for path in TEMPLATE_MANAGED:
            if not any(f == path or f.startswith(path + "/") for f in template_files):
                continue
            git(wt, "rm", "-r", "-q", "--ignore-unmatch", "--", path)
            git(wt, "checkout", sha, "--", path)
        for path, folder in TEMPLATE_IF_MISSING.items():
            if path in template_files and not list_files(wt, "HEAD", folder.rstrip("/")):
                git(wt, "checkout", sha, "--", path)

        if not git(wt, "diff", "--cached", "--name-only").strip():
            print("✅ Project already matches the template – nothing to do.")
            return
        print(git(wt, "diff", "--cached", "--stat"))
        git(wt, "commit", "-q", "-m", f"Sync vault template ({sha[:8]})")
        git(wt, "push", "-q", "-f", "origin", SYNC_BRANCH)

    url = web_url(repo)
    print(f"✅ Pushed {SYNC_BRANCH}. Review and merge it into {MAIN}:")
    if url:
        print(f"   {url}/-/merge_requests/new?merge_request[source_branch]={SYNC_BRANCH}")
    print("   Interview branches created after the merge include the update.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="local clone of the vault repo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--only", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("review")
    p.add_argument("interview")
    p = sub.add_parser("collect")
    p.add_argument("interview")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("sync-template")
    p.add_argument("--template", required=True, help="URL (or path) of the vault-template repo")
    p.add_argument("--ref", default=MAIN, help="template branch (default: main)")
    args = parser.parse_args()

    if args.cmd == "setup":
        cmd_setup(args.repo, args.only, args.dry_run)
    elif args.cmd == "review":
        cmd_review(args.repo, args.interview)
    elif args.cmd == "collect":
        cmd_collect(args.repo, args.interview, args.force)
    else:
        cmd_sync_template(args.repo, args.template, args.ref)


if __name__ == "__main__":
    main()
