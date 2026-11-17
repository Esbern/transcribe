import argparse
from pathlib import Path
import subprocess
import shutil

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
            shutil.copy(template_path, target)
            print(f"  ✅ Created notebooks/{target.name}")

    # 2. Create config.yml Template
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        create_config_yaml(config_file, project_name)

    print("\n🎉 Setup complete! Next steps:")
    print("1. Open 'config.yml' and add your HF_TOKEN (if needed).")
    print(f"2. Run 'python -m qualivault.cli --notebook {project_name}' to start.")

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