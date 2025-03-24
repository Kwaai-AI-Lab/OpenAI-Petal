#!/bin/bash

# KwaaiNet for Mac One-Step Installer
# This script handles Python installation and environment setup before installing KwaaiNet

set -e  # Exit on error

echo "=========================================================="
echo "KwaaiNet for Mac - One-Step Installer"
echo "=========================================================="

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This installer is only for macOS systems."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew
install_homebrew() {
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH if needed
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "Installing Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$(uname -m).sh"
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    
    curl -fsSL $MINICONDA_URL -o $MINICONDA_INSTALLER
    bash $MINICONDA_INSTALLER -b -p $HOME/miniconda
    rm $MINICONDA_INSTALLER
    
    # Add Miniconda to PATH
    if [[ -z "${CONDA_PREFIX}" ]]; then
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.zprofile
        export PATH="$HOME/miniconda/bin:$PATH"
    fi
    
    # Initialize conda for shell
    $HOME/miniconda/bin/conda init "$(basename "${SHELL}")"
}

# Install Xcode Command Line Tools if needed
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "Please wait for Xcode Command Line Tools to finish installing, then press Enter to continue..."
    read -p "Press Enter to continue..."
fi

# Install Homebrew if needed
if ! command_exists brew; then
    echo "Homebrew not found. Installing..."
    install_homebrew
else
    echo "Homebrew already installed."
fi

# Install Miniconda if needed
if ! command_exists conda; then
    echo "Conda not found. Installing Miniconda..."
    install_miniconda
    
    # Source conda after installing
    if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
        . "$HOME/miniconda/etc/profile.d/conda.sh"
    fi
else
    echo "Conda already installed."
    
    # Source conda
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
    fi
fi

# Create environment and install KwaaiNet
echo "Setting up KwaaiNet environment..."
if ! conda info --envs | grep -q kwaainet; then
    conda create -y -n kwaainet python=3.10
else
    echo "Using existing kwaainet environment..."
fi

# Activate the environment
conda activate kwaainet || source activate kwaainet

# Install the package directly from your URL
echo "Installing KwaaiNet for Mac..."
pip install --no-cache-dir https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.1.0.tar.gz

# Run initial setup
echo "Running initial setup..."
kwaainet setup

echo "=========================================================="
echo "KwaaiNet for Mac has been successfully installed!"
echo ""
echo "To activate the environment and use KwaaiNet, run:"
echo "  conda activate kwaainet"
echo ""
echo "To start a KwaaiNet node:"
echo "  kwaainet start"
echo ""
echo "For more information, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"
echo "=========================================================="