#!/usr/bin/env python3
"""
Install essential packages for DharmaMind
"""
import subprocess
import sys

packages = [
    "fastapi",
    "uvicorn[standard]", 
    "redis",
    "fakeredis",
    "asyncpg",
    "sqlalchemy",
    "psutil"
]

def install_packages():
    """Install all required packages"""
    print("Installing DharmaMind requirements...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"SUCCESS: {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install {package}")
    
    print("Installation complete!")

if __name__ == "__main__":
    install_packages()
