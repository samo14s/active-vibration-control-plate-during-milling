"""
setup_workspace.py
==================
Setup the working directory by copying all Python modules to a flat structure.

This script:
1. Copies all .py files from subdirectories to a single working folder
2. Creates the figs_article_publication output directory
3. Verifies all imports work

Usage:
    python setup_workspace.py
"""

import os
import shutil
import sys

# Source directories
SOURCE_DIRS = [
    '01_core',
    '02_controllers',
    '03_analysis',
    '04_figures',
    '05_main',
]

# Target working directory
WORK_DIR = 'workspace'


def main():
    print("="*70)
    print(" SETUP WORKSPACE — Article Simulation Package ")
    print("="*70)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create workspace
    workspace_path = os.path.join(current_dir, WORK_DIR)
    if os.path.exists(workspace_path):
        print(f"\n[!] Workspace already exists: {workspace_path}")
        response = input("    Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("    Aborted.")
            return
        shutil.rmtree(workspace_path)
    
    os.makedirs(workspace_path)
    print(f"\n[1/3] Created workspace: {workspace_path}")
    
    # Copy all .py files from subdirectories
    total_files = 0
    print("\n[2/3] Copying source files:")
    
    for src_dir_name in SOURCE_DIRS:
        src_dir = os.path.join(current_dir, src_dir_name)
        if not os.path.exists(src_dir):
            print(f"    [!] Missing: {src_dir_name}")
            continue
        
        for filename in sorted(os.listdir(src_dir)):
            if filename.endswith('.py'):
                src = os.path.join(src_dir, filename)
                dst = os.path.join(workspace_path, filename)
                shutil.copy2(src, dst)
                total_files += 1
                print(f"    ✓ {src_dir_name}/{filename}")
    
    # Create output directory inside workspace
    os.makedirs(os.path.join(workspace_path, 'figs_article_publication'),
                exist_ok=True)
    
    # Test imports
    print(f"\n[3/3] Verifying imports ...")
    sys.path.insert(0, workspace_path)
    
    test_imports = [
        'kirchhoff_q4',
        'plate_model',
        'piezo_actuator',
        'milling_force',
        'newmark_solver',
        'lqg_controller',
        'darc_controller',
        'imc_lqg_controller',
        'fdm_stability',
    ]
    
    failed = []
    for mod in test_imports:
        try:
            __import__(mod)
            print(f"    ✓ {mod}")
        except Exception as e:
            print(f"    ✗ {mod}: {e}")
            failed.append(mod)
    
    print(f"\n{'='*70}")
    print(f" Setup complete: {total_files} files copied to '{WORK_DIR}/'")
    if failed:
        print(f" [!] {len(failed)} import(s) failed: {failed}")
    else:
        print(f" ✓ All imports verified")
    print(f"{'='*70}")
    
    print(f"\nNext steps:")
    print(f"  cd {WORK_DIR}")
    print(f"  python main_simulation.py              # Run main simulation")
    print(f"  python gen_article_complete_figures.py # Generate 14 figures")
    print(f"  python gen_geometry_figure.py          # Generate geometry")


if __name__ == '__main__':
    main()
