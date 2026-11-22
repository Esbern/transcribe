## 🔐 Phase 1: Security & Authentication

Standard passwords do not work for syncing. You must use a **Personal Access Token (PAT)**.

### 1. Generate your Token

1. Log in to **[git.ruc.dk](https://git.ruc.dk)**.
    
2. Go to **User Settings** (top right icon) → **Access Tokens**.
    
3. Click **Add new token**.
    
    - **Name:** `Obsidian-Laptop`
        
    - **Scopes:** Check **`write_repository`** ONLY.
        
4. Click **Create token** and **COPY IT IMMEDIATELY**.
    

### 2. Create your Login String

Combine your credentials into this format: `https://[USERNAME]:[TOKEN]@git.ruc.dk/[PROJECT-PATH].git`

- _Example:_ `https://j_doe:glpat-12345abcdef@git.ruc.dk/holmes/legendary_interview.git` ✅
    

---

## 🛠️ Phase 2: First-Time Setup

1. **Install Tools:** Download [Git](https://git-scm.com/) and [GitHub Desktop](https://desktop.github.com/).
    
2. **Clone the Vault:** In GitHub Desktop, go to `File` -> `Clone Repository` -> `URL`. Paste your **Login String**.
    
3. **Open Obsidian:** "Open folder as vault" and select the folder you just cloned.
    
4. **Configure Obsidian Git:** In Obsidian, press `Ctrl+P`, type `Obsidian Git: Edit remote URL`. Paste your Login String again to enable internal syncing.
    

---

## 🔄 Phase 3: The Daily Workflow (The "Safety First" Method)

**Crucial:** Do not open Obsidian until you have performed Step 1 in GitHub Desktop.

### 1. Start Your Session (In GitHub Desktop)

1. **Select Main:** Ensure the "Current Branch" is set to `main`.
    
2. **Fetch & Pull:** Click "Fetch origin" to get the latest data.
    
3. **Create Branch:** Click the Branch menu -> **New Branch**.
    
    - _Name format:_ `coding-[name]-[interview-id]` (e.g., `coding-jane-int01`).
        
4. **Publish:** Click **"Publish branch"** to send it to the server.
    
5. **Claim Task:** Go to the GitLab **Issue Board** and drag your interview to **"Doing."**
    

### 2. Prepare the File (In Obsidian)

_Now you can open Obsidian safely._

1. Right-click the file in `01_Raw_Transcripts` → **Make a copy**.
    
2. Drag the copy into `02_Coding_In_Progress` and rename it.
    

### 3. Coding & Audio

- **Audio:** Use the MS Teams link and **Media Extended** timestamps (`Ctrl+P`).
    
- **New Tags:** If you need a new tag, create a **New Issue** in GitLab using the `new_tag_proposal` template. Do not edit the Codebook file.
    

---

## 🚩 Phase 4: The Hand-off

1. **Sync work:** In Obsidian, run `Ctrl+P` → `Obsidian Git: Push`.
    
2. **Signal Done:** Go to the GitLab **Issue Board**. Drag your Issue to **"Review-Needed."**
    
3. **Comment:** Leave a comment: _"Finished coding in branch [your-branch-name]. @supervisor_name"_
    
4. **Close Obsidian:** Close the app to stop it from making more background changes.
    
5. **Back to Main:** In **GitHub Desktop**, switch your branch back to `main`. If it asks what to do with changes, select **"Discard changes"** (this cleans up the Obsidian noise).
    

---

## 🧹 Maintenance & Safety

### Cleaning up Branches

The supervisor deletes your branch on the server after merging. To clean your local list:

- In GitHub Desktop, go to `Data` -> `Prune online branches`.
    

### ⚠️ Important Safety Rules

- **GitHub Desktop First:** Always create your branch _before_ opening Obsidian.
    
- **The Baseline:** Never edit files in `01_Raw_Transcripts`.
    
- **Conflicts:** If you see a "Merge Conflict" error, **STOP**. Contact the supervisor immediately.
    

---

### Final Workflow Summary

|Action|Who|Where|
|---|---|---|
|**New Branch**|Student|**GitHub Desktop** (Before opening Obsidian)|
|**Pick Task**|Student|GitLab Issue Board (Move to `Doing`)|
|**Prepare File**|Student|Obsidian (Duplicate from `01_Raw`)|
|**Code & Sync**|Student|Obsidian (Push to Branch)|
|**Signal Done**|Student|GitLab Issue Board (Move to `Review-Needed`)|
|**Review & Merge**|Supervisor|GitLab (Merge & Move file to `03_Complete`)|