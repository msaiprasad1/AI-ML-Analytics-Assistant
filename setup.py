#!/usr/bin/env python3
"""
Setup script for Excel ML Chat Assistant
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install packages: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        print("Creating .env file...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("✓ Created .env file")
        print("⚠️  Please edit .env file and add your OpenAI API key")
    else:
        print("✓ .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "visualizations"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory}/ directory")
        else:
            print(f"✓ {directory}/ directory already exists")

def create_sample_data():
    """Create sample datasets"""
    if not os.path.exists("sample_housing_data.xlsx"):
        print("Creating sample datasets...")
        try:
            subprocess.check_call([sys.executable, "create_sample_data.py"])
            print("✓ Sample datasets created")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to create sample data: {e}")
    else:
        print("✓ Sample datasets already exist")

def main():
    """Main setup function"""
    print("🤖 Excel ML Chat Assistant Setup")
    print("=" * 40)
    
    check_python_version()
    install_requirements()
    create_env_file()
    create_directories()
    create_sample_data()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python main.py")
    print("3. Open: http://127.0.0.1:8000")
    print("\nSample queries to try:")
    print("- 'Run EDA on this dataset'")
    print("- 'Perform linear regression using Price as target'")
    print("- 'Show me feature correlations'")

if __name__ == "__main__":
    main()
