import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

def create_workspace(project_name):
    """
    Scaffolds a new QualiVault workspace in projects/<project_name>.
    Creates a .env file and a template Jupyter notebook.
    """
    cwd = Path.cwd()
    project_dir = cwd / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🤖 Initializing QualiVault Workspace in: {project_dir}")

    # 1. Create Notebooks Directory
    nb_dir = project_dir / "notebooks"
    nb_dir.mkdir(exist_ok=True)
    
    # 2. Create Preparation Notebook
    prep_nb_path = nb_dir / "01_Prepare_Audio.ipynb"
    if not prep_nb_path.exists():
        create_prepare_notebook(prep_nb_path)
        print("  ✅ Created template: notebooks/01_Prepare_Audio.ipynb")
    else:
        print("  ℹ️  Notebook already exists")

    # 3. Create Transcribe Notebook
    transcribe_nb_path = nb_dir / "02_Transcribe.ipynb"
    if not transcribe_nb_path.exists():
        create_transcribe_notebook(transcribe_nb_path)
        print("  ✅ Created template: notebooks/02_Transcribe.ipynb")
    else:
        print("  ℹ️  Notebook already exists")

    # 4. Create Export Notebook
    export_nb_path = nb_dir / "03_Export_Obsidian.ipynb"
    if not export_nb_path.exists():
        create_export_notebook(export_nb_path)
        print("  ✅ Created template: notebooks/03_Export_Obsidian.ipynb")
    else:
        print("  ℹ️  Notebook already exists")

    # 3. Create Setup Notebook (For Regex Testing)
    setup_nb_path = nb_dir / "00_Setup_and_Scan.ipynb"
    if not setup_nb_path.exists():
        create_setup_notebook(setup_nb_path)

    # 4. Create config.yml Template
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        create_config_yaml(config_file, project_name)

    print("\n🎉 Setup complete! Next steps:")
    print("1. Open 'config.yml' and add your HF_TOKEN (if needed).")
    print(f"2. Run 'python -m qualivault.cli --notebook {project_name}' to start.")

def create_prepare_notebook(path):
    """Copies the prepare audio notebook from templates."""
    template_path = Path(__file__).parent / "templates" / "01_Prepare_Audio.ipynb"
    
    if template_path.exists():
        shutil.copy(template_path, path)
        print("  ✅ Created prepare notebook from template")
    else:
        print(f"  ⚠️ Template not found at {template_path}")

def create_transcribe_notebook(path):
    """Copies the transcribe notebook from templates."""
    template_path = Path(__file__).parent / "templates" / "02_Transcribe.ipynb"
    
    if template_path.exists():
        shutil.copy(template_path, path)
        print("  ✅ Created transcribe notebook from template")
    else:
        print(f"  ⚠️ Template not found at {template_path}")

def create_export_notebook(path):
    """Copies the export notebook from templates."""
    template_path = Path(__file__).parent / "templates" / "03_Export_Obsidian.ipynb"
    
    if template_path.exists():
        shutil.copy(template_path, path)
        print("  ✅ Created export notebook from template")
    else:
        print(f"  ⚠️ Template not found at {template_path}")

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

def create_setup_notebook(path):
    """Copies the setup notebook from templates."""
    # Look for the template in the 'templates' folder relative to this script
    template_path = Path(__file__).parent / "templates" / "00_Setup_and_Scan.ipynb"
    
    if template_path.exists():
        shutil.copy(template_path, path)
        print("  ✅ Created setup notebook from template")
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

def main():
    parser = argparse.ArgumentParser(description="QualiVault CLI")
    parser.add_argument("--init", type=str, metavar="PROJECT_NAME", help="Initialize workspace in projects/<name>")
    parser.add_argument("--notebook", type=str, metavar="PROJECT_NAME", help="Start Jupyter Lab in project folder")
    args = parser.parse_args()

    if args.init:
        create_workspace(args.init)
    elif args.notebook:
        start_notebook(args.notebook)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()