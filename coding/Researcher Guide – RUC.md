# Researcher Guide – Running an Interview Project at RUC

*For RUC researchers who want to transcribe interviews with QualiVault and have student assistants validate and code them in Obsidian, using the shared GitLab group **interview-projects** on git.ruc.dk.*

Not at RUC? First read **[Researcher Guide – Own Infrastructure](Researcher%20Guide%20–%20Own%20Infrastructure.md)**. It shows how to build the same setup elsewhere, and then sends you back here.

---

## Overview

```
 Recordings ──► QualiVault (your computer) ──► Obsidian vault ──► GitLab project
  (audio)       transcribe + diarize           01_Raw_Transcripts   interview-projects/<project>
                                               compact_audio/          │
                                                                       ▼
                         03_Complete ◄── review + collect ◄── one branch per interview
                         (main)                                  (students validate + code)
```

| Phase | Who | Where | Time |
|---|---|---|---|
| 1. Access and data protection | Researcher | RUC / GitLab | days (paperwork) |
| 2. Transcription | Researcher | Own computer | hours of setup, then a night or weekend of computing |
| 3. GitLab project | Researcher | git.ruc.dk | ~1 hour |
| 4. Students | Researcher + students | Introduction session | ~1 hour |
| 5. Review and collect | Researcher | Own computer + GitLab | ongoing |

What is already shared in the group, so you don't need to build it:

- **`interview-projects/vault-template`**: Obsidian plugins and settings, GitLab issue templates, Templater scripts, the student laptop setup script, the **Student Guide** and the **Git Introduction**.
- **`coding/researcher/interview_branches.py`** in this repository: the tool that makes one branch per interview and brings finished work back.

---

## Phase 1 – Access and data protection

1. **Data protection.** Interviews are personal data. Before recording, make sure the project is registered and approved according to RUC's GDPR procedure, and that the consent forms allow transcription and analysis by student assistants. Everything in this workflow stays on your own computer and RUC's GitLab. No audio or text is sent to outside services.
2. **GitLab access.** Log in once at [git.ruc.dk](https://git.ruc.dk) with your RUC login. Then ask the owner of the `interview-projects` group to give you the **Maintainer** role in the group, or in a subgroup for your project.
3. **Your access token.** In GitLab: avatar → **Edit profile** → **Access tokens** → **Add new token**. Use scopes `read_repository` and `write_repository`, and set an expiry date. Copy it immediately, because GitLab shows it only once. Git over HTTPS accepts only this token, not your password.

---

## Phase 2 – Transcription with QualiVault

### 2.1 Install

You need a computer with an NVIDIA GPU (Windows with WSL2, or Linux). An Apple Silicon Mac or a CPU also works, but is much slower. Follow the installation in the main **README** of this repository: conda environment, `pip install -e .`.

Diarization (who speaks when) uses models from Hugging Face that require you to accept their conditions:

1. Create a free account at [huggingface.co](https://huggingface.co), and create a **read** token (Settings → Access Tokens).
2. While logged in, open [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the user conditions on both.

Only the models are downloaded. Your audio never leaves your computer.

### 2.2 Create the project

```bash
python -m qualivault.cli --init <Project>
```

This creates `projects/<Project>/` with `config.yml`, a `.env` and the notebooks.

- **`.env`**: put `HF_TOKEN=<your Hugging Face token>` here. The file is never committed.
- **Recordings**: ask for **one folder per interview, named `<name>_<id>`**, e.g. `Hansen_12`. Several files in one folder (a break, a new room) are fine: each file is processed separately and joined afterwards.

### 2.3 Configure `config.yml`

The most important settings:

| Setting | What to put |
|---|---|
| `paths.org_audio_folder` | Folder with the interview folders |
| `paths.flac_output_folder`, `paths.output_base_folder` | Working folders **outside** the project folder |
| `paths.obsidian_vault_folder` | Where the vault is built, e.g. `~/data/<project>/vault` |
| `paths.obsidian_notes_subfolder` | `01_Raw_Transcripts` |
| `paths.compact_audio_bitrate` | `32k` (speech; about 14 MB per hour) |
| `paths.compact_audio_in_git` | `true` (the recordings go into GitLab via Git LFS) |
| `transcription.language` | e.g. `da` |
| `transcription.topic_prompt` | A few sentences about the subject plus a list of words that will occur. This noticeably improves names and jargon. |
| `transcription.diarization.speaker_count_source` | `recipe`, so the `speakers` number you fill in below is used exactly |
| `text_corrections` | Words the AI consistently gets wrong: `wrong: right`. Fill this in later (see 5.4). |

### 2.4 Scan, check and run

1. Open the project (`python -m qualivault.cli --notebook <Project>`) and run **`00_Setup_and_Scan`**. It writes `processing_recipe.yaml` with one entry per interview, or per file for multi-file interviews.
2. In `processing_recipe.yaml`, fill in **`speakers`** (the number of people talking) for every entry. This matters a lot for getting the speaker labels right.
3. Start the unattended run:

   ```bash
   python -m qualivault.cli --run <Project>
   ```

   It converts, transcribes (resuming after interruptions, with retries), joins multi-file interviews, cleans the text, builds the small mp3 files and writes the vault. Expect several hours for a typical project: run it overnight or over a weekend, and keep the computer awake.
4. Skim a few notes in the vault. If whole passages are wrong, adjust `topic_prompt` and re-transcribe. If only some words are wrong, use `text_corrections` and `--export` (see 5.4).

---

## Phase 3 – The GitLab project

### 3.1 Create the project

In GitLab, inside **interview-projects**:

1. **New subgroup** named after your project, e.g. `hakan`, so you can manage members per project.
2. In the subgroup: **New project → Create blank project**, same name, visibility **Private**. **Untick** "Initialize repository with a README": the project must be empty.

The address becomes `https://git.ruc.dk/interview-projects/<project>/<project>.git`.

Put it and your token in the QualiVault project's `.env`, which the tools read:

```
RUC_GIT=https://git.ruc.dk/interview-projects/<project>/<project>
RUC_GIT_token=<your GitLab token>
```

### 3.2 First upload of the vault

**Order matters:** Git LFS and `.gitattributes` must be in place *before* the recordings are added, otherwise the audio goes into the normal history and makes the repository huge.

```bash
cd ~/data/<project>/vault            # paths.obsidian_vault_folder
git init -b main
git lfs install --local
printf 'compact_audio/*.mp3 filter=lfs diff=lfs merge=lfs -text\n' > .gitattributes
mkdir -p 02_Coding_In_Progress 03_Complete && touch 02_Coding_In_Progress/.gitkeep 03_Complete/.gitkeep
git add .gitattributes && git commit -m "Track recordings with Git LFS"
git add -A && git commit -m "Raw transcripts and recordings"
git lfs ls-files | head              # the mp3s must be listed here
git remote add origin https://git.ruc.dk/interview-projects/<project>/<project>.git
git push -u origin main              # user name: your GitLab user; password: your token
```

### 3.3 Add the shared template

```bash
python coding/researcher/interview_branches.py --repo ~/data/<project>/vault \
    --env-file projects/<Project>/.env \
    sync-template --template https://git.ruc.dk/interview-projects/vault-template
```

This pushes a `sync-template` branch. Open the link it prints, create the merge request and merge it. The vault now has the plugins, issue templates, Student Guide, Git Introduction and laptop setup script.

To read the template, the script uses the token in `templet_GIT_TOKEN` in the `.env` at the root of this repository. If you don't have one, ask the group owner for read access to `vault-template`.

### 3.4 Settings in GitLab

- **Protect `main`.** Settings → Repository → Protected branches: `main`, *Allowed to merge* and *Allowed to push*: **Maintainers**. Students can then never change `main`.
- **Labels** (Manage → Labels): `Doing`, `Review-Needed`, `Status::Stuck`, `Type::Codebook-Proposal`.
- **Issue board** (Plan → Issue boards): add lists for `Doing` and `Review-Needed`.
- **One issue per interview**, using the `default` template, titled e.g. `Interview_7`.
- **Codebook.** Add one note per tag in `Tags/`, e.g. `Tags/Safety/Trust.md`, through a merge request. Students propose new tags with the `new_tag_proposal` issue template and mention you in it, so you are notified. Filter the issue list on the `Type::Codebook-Proposal` label to see all proposals.
- **README.** Add a short README for students: what the project is, and the first-download command (see Hakan's README as an example).

### 3.5 One branch per interview

```bash
python coding/researcher/interview_branches.py --repo ~/data/<project>/vault \
    --env-file projects/<Project>/.env setup
```

This creates `interview-Interview_<n>` for every note in `01_Raw_Transcripts`. Each branch contains only that interview, its working copy in `02_Coding_In_Progress`, its recording and the shared files. The `setup-Interview_<n>` tag marks the starting point, so you can later see exactly what the student changed. Use `--dry-run` first, or `--only Interview_7` for a single interview.

---

## Phase 4 – Students

1. Students log in at git.ruc.dk once. Add them to the project as **Developer**: they can push to interview branches but not to `main`.
2. Hold an introduction session. The students follow the **Student Guide**, and the **Git Introduction** explains the ideas. Both are in the vault. Together you check:
   - disk encryption
   - the token
   - the first `git clone --branch interview-Interview_<n> …`
   - `bash .tools/student_laptop_setup.sh`
   - that the audio plays in Obsidian
3. Assign interviews through the issues on the board.

---

## Phase 5 – Review, collect and corrections

### 5.1 Review

When an issue moves to **Review-Needed**:

```bash
python coding/researcher/interview_branches.py --repo ~/data/<project>/vault --env-file projects/<Project>/.env review Interview_7
```

This lists what changed since setup and prints a GitLab compare link, where you read the corrections and codes line by line. It warns if files other than the working copy were changed. Obsidian settings changes are ignored.

### 5.2 Collect

```bash
python coding/researcher/interview_branches.py --repo ~/data/<project>/vault --env-file projects/<Project>/.env collect Interview_7
```

This copies the finished file into `03_Complete/` on a `collect-Interview_7` branch. Open the merge request it links to and merge it. Afterwards you may delete the `interview-Interview_7` branch; the setup tag keeps the history.

> ⚠️ **Never merge an `interview-…` branch into `main`** with GitLab's merge button. The branch records the removal of all the other interviews, so merging it would delete them from `main`.

### 5.3 Problems

- **Conflict.** A student who reports a conflict has usually worked on two computers. Look at the branch in GitLab before doing anything, because every version is kept.
- **Audio doesn't play.** Git LFS is missing on the laptop: run the setup script again.
- **Lost laptop.** Remove the student from the project and ask them to revoke their token.

### 5.4 Correcting recurring errors (re-export)

As students report words the AI gets wrong, add them to `text_corrections` in `config.yml`, then:

```bash
python -m qualivault.cli --export <Project>      # minutes: no re-transcription
cd ~/data/<project>/vault && git add 01_Raw_Transcripts && git commit -m "Apply text corrections" && git push
```

Interview branches are snapshots of `main`, so existing branches do **not** get the corrected text:

- **No student has started yet**: delete the branches and `setup-*` tags, then run `setup` again.
- **Students have started**: leave their branches alone, and only (re)create branches for interviews nobody has opened (`setup --only …`).

So collect the misheard words from the first one or two interviews *before* handing out the rest.

---

## Keeping the template up to date

Improvements to the Student Guide, plugins or scripts are made once, in `vault-template`, through a merge request. Each project picks them up with `sync-template` (3.3). Branches created after that include the update.
