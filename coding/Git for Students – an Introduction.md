# Git for Students – an Introduction

*For student assistants who correct and code interview transcripts. You don't need to know anything about programming.*

You'll be working with a tool called **Git**. You will hardly ever "use" it directly, because Obsidian does most of the work for you. Still, a little understanding of what is going on helps a lot. It explains why the Student Guide asks you to do things in a particular order, and why you shouldn't panic if something looks strange.

This introduction explains the ideas. The **Student Guide** tells you exactly which buttons to press.

---

## 1. What problem does Git solve?

Imagine five people correcting and coding twenty interview transcripts at the same time, as shared Word files in a cloud folder. You probably know what happens:

- Two people open the same file, and one person's work overwrites the other's.
- Nobody can tell what the AI originally wrote and what a person changed.
- Somebody deletes a paragraph by accident, and it's gone.
- Files multiply: `Interview_7_final.docx`, `Interview_7_final_v2_checked.docx` …

Git was made for exactly this. Programmers use it to work on the same text files without destroying each other's work, and our transcripts are just text files too. Git gives us:

- **A complete history.** Every saved change is kept, with who made it and when. Nothing is ever really lost.
- **A clean original.** The AI transcript is never touched, so your corrections can always be compared with it.
- **Separate workspaces.** You work on your own copy of your own interview, so nobody can overwrite your work, and you can't overwrite anybody else's.
- **Review.** The researcher can see exactly which words you changed and which codes you added, line by line.

For qualitative research this is a quality guarantee: we can document how the data went from audio to coded text.

---

## 2. The six ideas you need

### Repository – the shared archive

A **repository** (or "repo") is a project folder with a memory. It holds all the interview transcripts, the recordings, the codebook and the Obsidian settings, and it remembers every version of every file.

The main copy lives on the university's server, **GitLab** (git.ruc.dk). That server is inside RUC, so the interviews never go to an outside company. You download a copy to your laptop, and that copy is your Obsidian vault.

> **Think of it as** an archive at the university. You get your own copy to work on at home, and you hand your changes back at the counter.

### Commit – a saved version with a label

In Word, "Save" overwrites the file. In Git, a **commit** is a snapshot of your changes that is added to the history *on top of* the earlier versions. Each commit records:

- what was changed (exactly which lines)
- who made the change
- when it was made
- a short message describing it

You don't have to write these messages yourself. Obsidian does it for you automatically, with a message like `vault backup: 2026-11-24 14:02`.

> **Think of it as** an archivist's log: "On 24 November at 14:02, Maria changed these three lines."

### Push and pull – upload and download

Your laptop and the server each have a copy of the repository. They don't stay in step by themselves:

- **Push** sends your commits *up* to the server.
- **Pull** fetches other people's commits *down* to your laptop.

Obsidian does both automatically while you work. It pulls when it starts, and it commits and pushes about every two minutes after you stop typing. The one thing you must remember is the last upload when you stop for the day. That's why the guide says:

> **Always finish with `Obsidian Git: Commit-and-sync and then close Obsidian`.**
> If you just close the window, your last changes stay on your laptop and the researcher can't see them.

### Branch – your own version of the project

A **branch** is a separate line of versions. The project has:

- **`main`**: the complete archive with all interviews. Only the researcher changes it.
- **`interview-Interview_7`**, **`interview-Interview_8`**, …: one branch per interview. On such a branch, the vault shows *only that interview*: the original, your working copy, its recording and the codebook.

When you "switch branch" in Obsidian, the files in your vault change to match that branch. This looks like files appear and disappear, but nothing is deleted. Each branch's files are kept safely on the server, and switching only changes which ones are on your laptop.

> **Think of it as** being handed a folder containing just one interview, instead of the whole filing cabinet. You work in your folder, and when you're done, the researcher files your finished copy in the cabinet.

That's also why you never need to worry about destroying `main`. You can't reach it from your branch, and the server protects it.

### Merge and conflict – when two changes meet

**Merging** means combining two lines of changes. Usually Git does this by itself without asking. If two people changed the *same line* in different ways, Git can't know which version is right, and it reports a **conflict**.

In our setup conflicts are rare, because only you work on your interview branch. They can still happen if the same branch was used on two computers, or if the researcher updated a file you had also changed.

> **If you ever see the word "conflict" or "merge": stop, don't click anything else, and contact the researcher.** Nothing is lost, but sorting it out is the researcher's job.

### Git LFS – the large audio files

Git is very good with text and not so good with large files like recordings. **Git LFS** (Large File Storage) is an add-on that stores the audio separately and downloads only the recordings for the interview you're working on.

If a recording won't play, or the audio file is tiny (about 1 KB), LFS is probably missing. Contact the researcher.

---

## 3. How a correction travels

What happens when you correct a misheard word:

```
 Your laptop (Obsidian vault)                      GitLab server (git.ruc.dk)
 ───────────────────────────                       ──────────────────────────
 1. You change "bandet" to "banderne"
    in 02_Coding_In_Progress/Interview_7.md
                │
                ▼ (Obsidian, automatically after ~2 minutes)
 2. Commit: "vault backup: 14:02"
                │
                ▼ (push)
                └──────────────────────────────►   3. Saved on branch
                                                      interview-Interview_7
                                                              │
                                                              ▼
                                                   4. Researcher reviews the
                                                      changes and files the
                                                      finished copy in main /
                                                      03_Complete
```

When the researcher reviews your work, they see a **diff**. A diff is the list of changed lines, with the old line marked `-` and the new line marked `+`. For example (invented text):

```diff
 ### SPEAKER_01 [00:12:40](compact_audio/Interview_7.mp3#t=760)
-Der er mange af de unge fra bandet, der hænger ud ved kiosken om aftenen.
+Der er mange af de unge fra banderne, der hænger ud ved kiosken om aftenen. #tryghed
```

This is why we work in Markdown text files rather than Word: Git can show exactly which word changed. It's also why **the original in `01_Raw_Transcripts` must never be edited**, because it's the baseline that your corrections are compared with.

---

## 4. What this means for your daily work

| You do… | What happens in Git |
|---|---|
| Open Obsidian | **Pull**: you get the latest version from the server |
| Switch to your interview's branch | Your vault now shows only that interview |
| Correct and code in `02_Coding_In_Progress` | Changes are saved on your laptop |
| Pause typing for ~2 minutes | **Commit + push**: your work is uploaded to your branch |
| Run *Commit-and-sync and then close Obsidian* | The last changes are uploaded, then Obsidian closes |
| Drag the issue to *Review-Needed* | The researcher knows your branch is ready to review |

### Good habits

- **Start with a pull, end with commit-and-sync.** Obsidian does the first one by itself, and you must do the second one.
- **Work on one laptop at a time.** If you used another computer, finish there with *Commit-and-sync* before you open Obsidian on this one.
- **Only change files in `02_Coding_In_Progress`.** Everything else is shared material.
- **Don't rename or move files.** Git sees that as "deleted one file, created another", which makes review harder.
- **Don't worry about settings changes.** Obsidian may record small changes to its own settings files (in `.obsidian`). That's harmless and ignored.

---

## 5. Frequently asked questions

**Can I break something?**
Not really. Every version is kept on the server, and `main` is protected. The worst realistic case is a conflict, which the researcher can sort out.

**I made a mistake and want an earlier version back.**
Don't try to fix it with Git yourself. Tell the researcher which file and roughly when. Because every commit is kept, earlier versions can always be recovered.

**Files disappeared from my vault!**
You probably switched branch. Each interview branch only shows its own interview. The other files are safe on the server.

**I closed Obsidian without commit-and-sync.**
Nothing is lost: the changes are still on your laptop. Open Obsidian again and run *Commit-and-sync and then close Obsidian*.

**Obsidian shows an error about "authentication" or "403".**
Your token has probably expired or was typed wrong. Contact the researcher, and never send the token by mail or chat.

**Why not just use OneDrive or SharePoint?**
They don't show *what* changed inside a text, they handle simultaneous editing badly, and they can't keep the original and the corrected version neatly apart. For research data we also want to control exactly where the files are stored: on RUC's own server.

---

## 6. Mini dictionary

| Word | Meaning |
|---|---|
| **Git** | The program that keeps track of versions |
| **GitLab** | RUC's server (git.ruc.dk) where the shared copy lives, plus a website with issues and the board |
| **Repository / repo** | The project folder with its full history |
| **Clone** | Download a repository for the first time (you do this once) |
| **Commit** | A saved snapshot of changes, with author, time and message |
| **Push** | Upload your commits to the server |
| **Pull** | Download other people's commits from the server |
| **Sync** | Pull and push in one go |
| **Branch** | A separate line of versions; here, one per interview |
| **main** | The complete, protected main branch |
| **Merge** | Combine two lines of changes |
| **Conflict** | Two changes to the same line that Git can't combine by itself, so a person must decide |
| **Diff** | The list of lines that changed (old `-`, new `+`) |
| **Git LFS** | Add-on that stores and downloads the large audio files |
| **Token** | A personal password for Git, made in GitLab, that you can revoke at any time |
| **Issue** | A task or question on GitLab; each interview has one on the Issue Board |
