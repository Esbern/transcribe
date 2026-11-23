# Researcher Guide – Building Your Own Setup

*For researchers outside RUC, or at RUC but outside the `interview-projects` group, who want the same workflow: QualiVault transcription, an Obsidian vault, and student assistants who validate and code one interview at a time on their own Git branch.*

At RUC most of this already exists: use **[Researcher Guide – RUC](Researcher%20Guide%20–%20RUC.md)** instead. This guide builds the shared pieces once. After that, every project follows the RUC guide from Phase 2 onwards, with your own server address in place of `git.ruc.dk/interview-projects`.

---

## What you are building

| Piece | What it is | Built |
|---|---|---|
| **Git server with LFS** | Stores the vaults, recordings, branches and issues | once (usually exists already) |
| **Group** | A GitLab group that holds the template and all projects | once |
| **Vault template** | A repository with the shared Obsidian setup, issue templates, scripts and guides | once, then maintained |
| **QualiVault + `interview_branches.py`** | This repository: transcription and branch tooling | install per researcher |
| **Project** | One repository per study, made from the template | per study |

---

## Step 1 – Choose the Git server

The workflow needs **GitLab**: issue templates, issue boards, protected branches, merge requests, and **Git LFS** for the recordings.

- **Your institution's own GitLab (recommended).** The data stays on servers your institution controls, which makes the GDPR assessment simple. Ask IT to confirm that **Git LFS is enabled** and that projects may hold a few GB: at 32 kbit/s, recordings take about 14 MB per hour, plus the text.
- **gitlab.com.** This works technically, but the interviews are then stored by a third party. Check with your data protection officer whether a data processing agreement and the storage location are acceptable *before* uploading anything.
- **GitHub or others.** Possible, but the issue templates, board and guides assume GitLab and would need adapting.

Everyone (researchers and students) logs in with **personal access tokens** (scopes `read_repository` and `write_repository`), because Git over HTTPS does not accept passwords.

---

## Step 2 – Create the group

Create a group, e.g. `interview-projects`, with visibility **Private**. You are its owner. Researchers who will run projects get the **Maintainer** role (in the group, or in a subgroup per project). Students are only ever added to individual projects.

---

## Step 3 – Build the vault template

Create a private project `vault-template` in the group, clone it, and add the files below. Commit through a merge request, like everything else.

### 3.1 Git files

**`.gitattributes`**: recordings go to Git LFS:

```
compact_audio/*.mp3 filter=lfs diff=lfs merge=lfs -text
compact_audio/*.m4a filter=lfs diff=lfs merge=lfs -text
compact_audio/*.wav filter=lfs diff=lfs merge=lfs -text
```

**`.gitignore`**: share the plugin setup, ignore personal Obsidian files:

```
# System files
.DS_Store
Thumbs.db
.trash/

# Obsidian: share the plugin setup, ignore personal/noisy files.
# NB: ".obsidian/*" (not ".obsidian") – otherwise the "!" lines below cannot re-include anything.
.obsidian/*
!.obsidian/app.json
!.obsidian/core-plugins.json
!.obsidian/community-plugins.json
!.obsidian/plugins/
```

### 3.2 Obsidian plugins and settings

1. Open the cloned `vault-template` folder as a vault in Obsidian.
2. Settings → Community plugins → enable community plugins, then install and enable:
   - **Obsidian Git**: saving and syncing
   - **Media Extended**: timestamps that play the recording
   - **Templater**: the helper scripts
3. Configure **Obsidian Git**. The resulting `.obsidian/plugins/obsidian-git/data.json` should contain:

   ```json
   {
     "autoSaveInterval": 2,
     "autoBackupAfterFileChange": true,
     "autoPullOnBoot": true,
     "pullBeforePush": true,
     "disablePush": false,
     "syncMethod": "merge"
   }
   ```

   This means: commit and sync 2 minutes after typing stops, and pull when Obsidian starts. Obsidian Git does *not* sync when the window is simply closed, which is why the Student Guide asks students to finish with **Commit-and-sync and then close Obsidian**.
4. Configure **Templater**: template folder `Templates`.
5. Commit `.obsidian/app.json`, `core-plugins.json`, `community-plugins.json` and the `plugins/` folder. The `.gitignore` above ensures that only these are shared.

To update a plugin later, replace its `main.js`, `manifest.json` and `styles.css` with the files from the plugin's latest release, and commit.

### 3.3 Issue templates (`.gitlab/issue_templates/`)

**`default.md`**: one issue per interview:

```markdown
### 📝 Interview
- **Interview:** `Interview_<n>`
- **Branch:** `interview-Interview_<n>`
- **Assigned to:** (student)

### ❓ Questions or problems
(Describe them here and mention the researcher.)

### ✅ Checklist
- [ ] Transcript validated against the audio
- [ ] Interview coded
- [ ] Ran **Obsidian Git: Commit-and-sync and then close Obsidian**
- [ ] Moved this issue to **Review-Needed**
```

**`new_tag_proposal.md`**: students propose codebook changes instead of editing the codebook:

```markdown
### 🏷️ New Tag Proposal
- **Proposed Tag Name:** `#category/new-tag`
- **Interview ID:** (e.g., Interview_04)
- **Timestamp:** (e.g., [12:45])

### 💡 Context/Definition
> [Write definition here]

### 📍 Example Quote
"..."

---
**Supervisor Action:** - [ ] Approved (Add to Master Codebook)
- [ ] Rejected (Reason in comments)

/label ~"Type::Codebook-Proposal"
/assign @<your-gitlab-user>
```

### 3.4 Scripts, codebook and guides

| Path in the template | Source in this repository | Adapt |
|---|---|---|
| `Templates/Clean_Transcript_Selection.md` | `coding/Scripts/Clean Transcript Selection.md` | – |
| `.tools/student_laptop_setup.sh` | `coding/researcher/student_laptop_setup.sh` | – |
| `Student Guide.md` | `coding/Student Guide.md` | server address in the clone command and token steps |
| `Git Introduction.md` | `coding/Git for Students – an Introduction.md` | server name (git.ruc.dk) |
| `Tags/README.md` | short note: one note per tag in `Tags/<Category>/<Tag>.md`; only the researcher edits it; students propose tags with the issue template | – |
| `README.md` | what the template is and how projects get it (Step 5) | – |

### 3.5 Check

Clone the template into a fresh folder, open it in Obsidian, and confirm that the three plugins are active without further setup (after "Trust author and enable plugins").

---

## Step 4 – Install the researcher tools

On the computer that will transcribe, install QualiVault as described in the main **README** and in Phase 2 of the RUC guide.

`coding/researcher/interview_branches.py` works with any Git server over HTTPS. It reads tokens from `.env` files:

| Variable (default name) | Used for | Where |
|---|---|---|
| `RUC_GIT_token` | the project repository | `projects/<Project>/.env`, passed with `--env-file` |
| `templet_GIT_TOKEN` | the vault-template repository | `.env` at the root of this repository |

If you prefer other names, pass `--token-var` and `--template-token-var`. The tokens are sent only to their own repository's address and are never saved in Git's configuration.

---

## Step 5 – Run projects

From here, follow **[Researcher Guide – RUC](Researcher%20Guide%20–%20RUC.md)** from **Phase 2**, replacing:

| In the RUC guide | For you |
|---|---|
| `https://git.ruc.dk` | your GitLab address |
| `interview-projects` | your group |
| `interview-projects/vault-template` | your template project |
| "ask the group owner" | you are the group owner |
| RUC's GDPR procedure | your institution's procedure |

The steps that use your own pieces are:

- **3.3 Add the shared template**: `sync-template --template <your template URL>` copies the template into the new project as a merge request.
- **3.4 Settings in GitLab**: protected `main`, labels (`Doing`, `Review-Needed`, `Status::Stuck`, `Type::Codebook-Proposal`), the issue board and one issue per interview.

---

## Checklist

- [ ] Git server with LFS confirmed, and data protection approved
- [ ] Private group created; researchers are Maintainers
- [ ] `vault-template` with `.gitattributes`, `.gitignore`, `.obsidian` (3 plugins + settings), issue templates, `Templates/`, `.tools/`, Student Guide, Git Introduction, `Tags/README.md`
- [ ] Template opens in a fresh Obsidian with the plugins active
- [ ] QualiVault installed, Hugging Face conditions accepted
- [ ] Tokens in `.env` files (never committed)
- [ ] First project made with the RUC guide, Phases 2–5
