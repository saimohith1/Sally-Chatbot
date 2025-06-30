#!/bin/bash

set -e

echo "Updating package lists..."
sudo apt update

echo "Installing Python 3.11+..."
sudo apt install -y python3.11 python3.11-venv python3.11-distutils

echo "Installing pip for Python 3.11..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

echo "Installing UV (>=0.7)..."
python3.11 -m pip install "uv>=0.7"

echo "Installing JupyterLab (>=4.2) and Streamlit (>=1.40)..."
python3.11 -m uv pip install "jupyterlab>=4.2" "streamlit>=1.40"

echo "Installing Git (>=2.0)..."
sudo apt install -y git

echo "Installing VS Code (>=1.99)..."
wget -qO- https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64 > vscode.deb
sudo apt install -y ./vscode.deb
rm vscode.deb

echo "All tools installed! Verifying versions:"
python3.11 --version
uv --version
jupyter-lab --version
streamlit --version
git --version
code --version

echo "All done!"