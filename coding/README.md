# Coding interviews with students – guides and tools

After QualiVault has transcribed the interviews, the Obsidian vault goes into a GitLab project. There, student assistants validate and code one interview at a time, each on their own Git branch.

## Where to start

| You are… | Read |
|---|---|
| A researcher at RUC (GitLab group `interview-projects`) | [Researcher Guide – RUC](Researcher%20Guide%20–%20RUC.md) |
| A researcher elsewhere, or building the setup from scratch | [Researcher Guide – Own Infrastructure](Researcher%20Guide%20–%20Own%20Infrastructure.md), then the RUC guide |
| A student assistant | [Student Guide](Student%20Guide.md) and [Git Introduction](Git%20for%20Students%20–%20an%20Introduction.md). Both are also in every project vault. |

## Contents

| Path | What |
|---|---|
| `researcher/interview_branches.py` | Researcher tool: `setup` one branch per interview, `review`, `collect` finished work, `sync-template` |
| `researcher/student_laptop_setup.sh` | Run once on each student laptop (Git LFS, name, removing old audio) |
| `Scripts/` | Templater scripts used in the vault |
| `Intro.md` | Short presentation of the approach |
| `Collaborative QDA with Obsidian & GitLab.md` | Earlier version of the workflow (GitHub Desktop), replaced by the guides above |

At RUC, the master copies of the student-facing files live in `interview-projects/vault-template`. The copies here are for setups outside RUC: when you change one, change the other too.
