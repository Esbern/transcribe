# Student Guide – Coding the Hakan Interviews

You will validate and code interview transcripts in **Obsidian**. Your work is saved to the
university's GitLab server (**git.ruc.dk**). You work on **one interview at a time**: your
vault only contains that interview, its recording and the codebook.

---

## Part 1 – One-time setup

*We do this together at the introduction session.*

### 1. Protect your laptop

The interviews contain personal information about real people. Before you start:

- **Disk encryption is on** – Mac: *FileVault*. Windows: *BitLocker* / *Device encryption*.
- **Screen locks automatically** after at most 5 minutes.

### 2. Install the programs

- **Obsidian** – [obsidian.md](https://obsidian.md)
- **Git** – Windows: [git-scm.com](https://git-scm.com) (includes Git LFS, which downloads the audio).
  Mac: we install Git and Git LFS together at the session.

### 3. Log in to GitLab

1. Go to **[git.ruc.dk](https://git.ruc.dk)** and log in with your normal RUC login.
2. Send your **GitLab username** (top-right icon) to the researcher. You will then be given access
   to the Hakan project.

### 4. Create your access token

Obsidian uses a *token* instead of your password.

1. In GitLab: top-right icon → **Edit profile** → **Access tokens** → **Add new token**.
2. Fill in exactly:

   | Field | Value |
   |---|---|
   | Token name | `obsidian-hakan` |
   | Expiration date | the end date of your contract |
   | Scopes | tick **only** `write_repository` |

3. Click **Create token** and **copy it right away** – GitLab shows it only once.
   Keep it somewhere safe (e.g. a password manager). **Never** send it by e-mail or chat.

### 5. Download your first interview

The researcher tells you which interview to start with, e.g. `Interview_7`.

1. Open **Terminal** (Mac) or **Git Bash** (Windows).
2. Paste this, replacing `Interview_7` with your interview:

   ```bash
   git clone --branch interview-Interview_7 https://git.ruc.dk/interview-projects/hakan/hakan.git Hakan
   ```

3. When asked, type your **GitLab username**, and paste your **token** as the password.
   (Nothing appears while you paste – that is normal.) Your computer remembers it.
4. The researcher then runs a few settings commands with you, including your name for the history.

### 6. Open the vault in Obsidian

1. Obsidian → **Open folder as vault** → choose the `Hakan` folder.
2. When Obsidian asks, click **Trust author and enable plugins**.

---

## Part 2 – For each interview

### Start

1. **GitLab Issue Board:** find the issue for your interview and drag it to **Doing**.
2. **Obsidian:** press `Ctrl+P` (Windows) / `Cmd+P` (Mac) and run:
   - `Obsidian Git: Pull`
   - `Obsidian Git: Switch branch` → choose `interview-<your interview>`, e.g. `interview-Interview_7`
3. Your vault now shows only this interview:

   | Folder | What it is | May you change it? |
   |---|---|---|
   | `01_Raw_Transcripts` | The original AI transcript – for looking back | ❌ No |
   | `02_Coding_In_Progress` | **Your working copy** | ✅ Yes |
   | `compact_audio` | The recording | ❌ No |
   | Codebook / tags | The agreed tags | ❌ No |

### Step 1 – Validate (correct the transcript)

Work in the file in **`02_Coding_In_Progress`**.

- Click a timestamp, e.g. `[00:04:20]`, to play the recording from that point.
- Correct words the AI misheard, and wrong speaker labels.
- Yellow **⚠ Validation Issues** boxes are places the AI thinks may be wrong. Check each one,
  fix the text if needed, then **delete the box**.
- Unsure what was there before you changed it? Open the same interview in `01_Raw_Transcripts`.

### Step 2 – Code

- Tag passages using the codebook.
- **Need a new tag?** Use it right away with the prefix `#new/`, e.g. `#new/trust`, **and** create a
  GitLab issue with the `new_tag_proposal` template. Do not edit the codebook yourself.

### Saving

Your work is saved and uploaded automatically while Obsidian is open and **when you close
Obsidian**. Close Obsidian when you stop for the day.

### Finish

1. Close Obsidian (this uploads your last changes).
2. **GitLab Issue Board:** drag your issue to **Review-Needed** and comment *"Finished"*.
3. When you get your next interview, start again at **Part 2 – Start**. The previous interview and
   its recording disappear from your laptop automatically.

---

## ⚠️ Rules

- **Only edit the file in `02_Coding_In_Progress`.**
- **Never** share your token or the vault folder, and never copy recordings elsewhere.
- **Error message about "conflict" or "merge"?** Stop, don't click anything else, and contact
  the researcher.
- **Audio won't play?** Contact the researcher – Git LFS is probably missing.
- **Lost or stolen laptop?** Tell the researcher immediately, and delete your token in GitLab
  (**Edit profile → Access tokens → Revoke**).
