#!/usr/bin/env python3
"""
Automated environment setup for StrokePredictorNet.
Checks dependencies, creates directories, and verifies installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def check_python_version():
    """Check if Python version is 3.8+"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Detected Python version: {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python {version_str} is not compatible. Need Python 3.8+")
        return False

def create_directory_structure():
    """Create project directory structure"""
    print_header("Creating Directory Structure")
    
    directories = [
        "config",
        "scripts",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/visualization",
        "src/utils",
        "data/raw",
        "data/processed",
        "data/splits",
        "data/synthetic",
        "data/.cache",
        "outputs/checkpoints",
        "outputs/logs/tensorboard",
        "outputs/predictions",
        "outputs/figures",
        "notebooks",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created {directory}/")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/inference/__init__.py",
        "src/visualization/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print_success(f"Created {init_file}")

def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        print_success("Upgraded pip")
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_success("Installed all dependencies")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print_header("Verifying Installation")
    
    packages = {
        "torch": "PyTorch",
        "nibabel": "NiBabel (medical imaging)",
        "monai": "MONAI (medical AI toolkit)",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "tqdm": "tqdm"
    }
    
    all_installed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} is installed")
        except ImportError:
            print_error(f"{name} is NOT installed")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA/GPU Support")
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print_warning("CUDA is not available. Training will use CPU (slower)")
    except ImportError:
        print_warning("PyTorch not installed yet")

def create_gitignore():
    """Create .gitignore file"""
    print_header("Creating .gitignore")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Outputs
outputs/*
!outputs/.gitkeep

# Logs
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Cache
data/.cache/*
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print_success("Created .gitignore")

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print(f"{GREEN}StrokePredictorNet environment is ready!{RESET}\n")
    
    print(f"{BLUE}Next Steps:{RESET}")
    print("  1. Download ISLES 2022 dataset:")
    print(f"     {YELLOW}python scripts/download_isles.py --output data/raw/ISLES2022{RESET}\n")
    
    print("  2. Create synthetic clinical metadata:")
    print(f"     {YELLOW}python scripts/create_synthetic_clinical.py{RESET}\n")
    
    print("  3. Preprocess data:")
    print(f"     {YELLOW}python scripts/preprocess_data.py --config config/config.yaml{RESET}\n")
    
    print("  4. Train model:")
    print(f"     {YELLOW}python scripts/train.py --config config/config.yaml --device cpu{RESET}\n")
    
    print(f"{BLUE}Documentation:{RESET}")
    print("  - README.md: Project overview")
    print("  - docs/: Detailed documentation")
    print("  - notebooks/: Jupyter notebooks for exploration\n")

def main():
    """Main setup function"""
    print_header("StrokePredictorNet Environment Setup")
    print("Author: Jamil Hanouneh")
    print("Version: 1.0.0\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create .gitignore
    create_gitignore()
    
    # Ask user if they want to install dependencies
    response = input("\nInstall Python dependencies now? (y/n): ").strip().lower()
    
    if response == 'y':
        if install_dependencies():
            verify_installation()
            check_cuda()
        else:
            print_error("Dependency installation failed")
            sys.exit(1)
    else:
        print_warning("Skipped dependency installation")
        print("Run manually: pip install -r requirements.txt")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
