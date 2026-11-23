import argparse
from pathlib import Path
import subprocess

def create_workspace(project_name):
    """
    Scaffolds a new QualiVault workspace in projects/<project_name>.
    Copies config.yml and all template notebooks.
    """
    cwd = Path.cwd()
    project_dir = cwd / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🤖 Initializing QualiVault Workspace in: {project_dir}")

    # 1. Copy every template notebook (00_Setup ... 05_Export) into the project
    nb_dir = project_dir / "notebooks"
    nb_dir.mkdir(exist_ok=True)
    templates_dir = Path(__file__).parent / "templates"
    for template_path in sorted(templates_dir.glob("*.ipynb")):
        target = nb_dir / template_path.name
        if target.exists():
            print(f"  ℹ️  notebooks/{target.name} already exists")
        else:
            content = template_path.read_text(encoding="utf-8")
            target.write_text(content.replace("{{PROJECT_NAME}}", project_name), encoding="utf-8")
            print(f"  ✅ Created notebooks/{target.name}")

    # 2. Create config.yml Template
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        create_config_yaml(config_file, project_name)

    # 3. Secrets go in a git-ignored .env, not in config.yml
    env_file = project_dir / ".env"
    if not env_file.exists():
        env_file.write_text("# Hugging Face token for pyannote diarization (never commit this file)\nHF_TOKEN=\n", encoding="utf-8")
        print("  ✅ Created .env (add your HF_TOKEN here)")
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            ".env\n.ipynb_checkpoints/\n__pycache__/\n*.wav\n*.WAV\n*.mp3\n*.m4a\n*.flac\n", encoding="utf-8"
        )
        print("  ✅ Created .gitignore (excludes .env and audio)")

    print("\n🎉 Setup complete! Next steps:")
    print("1. Put your Hugging Face token in .env (HF_TOKEN=...).")
    print("2. In config.yml set paths.org_audio_folder and the output folders, language and topic_prompt.")
    print(f"3. Run 'python -m qualivault.cli --notebook {project_name}' and start with 00_Setup_and_Scan.")

def create_config_yaml(path, project_name):
    """Creates config.yml from template."""
    template_path = Path(__file__).parent / "templates" / "config.yml"
    
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace("{{PROJECT_NAME}}", project_name)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  ✅ Created config.yml from template")
    else:
        print(f"  ⚠️ Template not found at {template_path}")

def start_notebook(project_name):
    """Starts Jupyter Lab in the project directory."""
    cwd = Path.cwd()
    project_dir = cwd / "projects" / project_name
    
    if not project_dir.exists():
        print(f"❌ Project '{project_name}' not found in {cwd / 'projects'}")
        return
    
    print(f"🚀 Starting Jupyter Lab in {project_dir}...")
    try:
        subprocess.run(["jupyter", "lab"], cwd=project_dir)
    except KeyboardInterrupt:
        print("\n🛑 Jupyter Lab stopped.")
    except Exception as e:
        print(f"❌ Error starting Jupyter Lab: {e}")

def _load_project(project_name):
    import os
    import yaml

    project_dir = (Path.cwd() / "projects" / project_name).resolve()
    if not (project_dir / "config.yml").exists():
        raise SystemExit(f"❌ No config.yml in {project_dir}")
    os.chdir(project_dir)  # core.setup_environment reads ./config.yml
    with open(project_dir / "config.yml", encoding="utf-8") as f:
        return project_dir, yaml.safe_load(f)


def export_project(project_name):
    """Clean transcripts (+ text_corrections), build compact audio, write the Obsidian vault."""
    from qualivault.export_workflow import run_export

    project_dir, config = _load_project(project_name)
    run_export(project_dir, config)
    print("✅ Export done.")


def run_project(project_name, attempts=3):
    """Unattended: convert/analyse, transcribe (resumable, retried), join parts, export."""
    import warnings

    from qualivault.core import setup_environment
    from qualivault.pipeline import prepare_audio_and_analyze, transcribe_recipe
    from qualivault.state import RunState

    project_dir, config = _load_project(project_name)
    if config.get("suppress_warnings", True):
        warnings.filterwarnings("ignore")
    setup_environment()
    recipe_path = project_dir / "processing_recipe.yaml"

    prepare_audio_and_analyze(project_root=project_dir, config=config, recipe_path=recipe_path, resume=True)
    for attempt in range(1, attempts + 1):
        try:
            transcribe_recipe(project_root=project_dir, config=config, recipe_path=recipe_path, resume=True)
        except Exception as e:
            print(f"❌ Transcription attempt {attempt} crashed: {e}")
            continue
        state = RunState(RunState.default_path(project_dir)).load()
        failed = [k for k, iv in state.data.get("interviews", {}).items()
                  if iv.get("steps", {}).get("transcribe", {}).get("status") == "failed"]
        if not failed:
            break
        print(f"⚠️ Attempt {attempt}: {len(failed)} interviews failed ({failed}); retrying" if attempt < attempts else f"⚠️ Giving up on: {failed}")
    from qualivault.export_workflow import run_export

    run_export(project_dir, config)
    print("✅ Pipeline done.")


def main():
    parser = argparse.ArgumentParser(description="QualiVault CLI")
    parser.add_argument("--init", type=str, metavar="PROJECT_NAME", help="Initialize workspace in projects/<name>")
    parser.add_argument("--notebook", type=str, metavar="PROJECT_NAME", help="Start Jupyter Lab in project folder")
    parser.add_argument("--run", type=str, metavar="PROJECT_NAME",
                        help="Unattended: convert, transcribe (resume), join parts, clean, compact audio, Obsidian export")
    parser.add_argument("--export", type=str, metavar="PROJECT_NAME",
                        help="Only clean transcripts (+ text_corrections), compact audio and Obsidian export")
    args = parser.parse_args()

    if args.init:
        create_workspace(args.init)
    elif args.notebook:
        start_notebook(args.notebook)
    elif args.run:
        run_project(args.run)
    elif args.export:
        export_project(args.export)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()