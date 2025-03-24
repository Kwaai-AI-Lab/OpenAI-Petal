#!/bin/bash

# KwaaiNet for Mac - One-Step Installer
# This script handles the entire installation process for KwaaiNet on macOS

set -e  # Exit on error

echo "=========================================================="
echo "KwaaiNet for Mac - One-Step Installer"
echo "=========================================================="
echo "This installer will set up KwaaiNet for sharing compute on macOS"
echo "It includes Python setup, dependencies, and environment configuration"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ Error: This installer is only for macOS systems."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew
install_homebrew() {
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH if needed
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    echo "✅ Homebrew installed successfully"
}

# Function to install Miniconda
install_miniconda() {
    echo "🐍 Installing Miniconda..."
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
    
    echo "✅ Miniconda installed successfully"
}

# Install Xcode Command Line Tools if needed
if ! xcode-select -p &>/dev/null; then
    echo "🛠 Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "⏳ Please wait for Xcode Command Line Tools to finish installing, then press Enter to continue..."
    read -p "Press Enter to continue..."
    echo "✅ Xcode Command Line Tools installed"
else
    echo "✅ Xcode Command Line Tools already installed"
fi

# Install Homebrew if needed
if ! command_exists brew; then
    echo "🍺 Homebrew not found. Installing..."
    install_homebrew
else
    echo "✅ Homebrew already installed"
fi

# Install Miniconda if needed
if ! command_exists conda; then
    echo "🐍 Conda not found. Installing Miniconda..."
    install_miniconda
    
    # Source conda after installing
    if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
        . "$HOME/miniconda/etc/profile.d/conda.sh"
    fi
else
    echo "✅ Conda already installed"
    
    # Source conda
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
    fi
fi

# Create environment and install KwaaiNet
echo "⚙️ Setting up KwaaiNet environment..."
if ! conda info --envs | grep -q kwaainet; then
    conda create -y -n kwaainet python=3.10
    echo "✅ Created Python 3.10 environment for KwaaiNet"
else
    echo "✅ Using existing kwaainet environment"
fi

# Activate the environment
conda activate kwaainet || source activate kwaainet

# Clear cached versions of the package
echo "🧹 Clearing any cached versions of KwaaiNet..."
pip cache remove kwaainet-mac &>/dev/null || true
pip cache remove kwaainet_mac &>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Install the package directly from your URL
echo "📦 Installing KwaaiNet for Mac..."
pip install --no-cache-dir https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.1.0.tar.gz

# Create launcher script for one-step execution
echo "🚀 Creating launcher script..."
LAUNCHER_PATH="$HOME/.local/bin/kwaainet"
mkdir -p "$HOME/.local/bin"

cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first

# Find conda
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/opt/anaconda3"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/opt/miniconda3"
else
    echo "❌ Error: Could not find conda installation."
    exit 1
fi

# Source conda without changing the prompt
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the environment and run the command
# Use a different name for the command inside conda to avoid potential recursion
conda activate kwaainet && python -m kwaainet.runner "$@"
EOF

chmod +x "$LAUNCHER_PATH"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.$(basename $SHELL)rc
    export PATH="$HOME/.local/bin:$PATH"
fi

# Try to create system-wide symlink if possible
if [ -w "/usr/local/bin" ]; then
    echo "📌 Creating system-wide link in /usr/local/bin..."
    ln -sf "$LAUNCHER_PATH" /usr/local/bin/kwaainet
fi

# Run initial setup
echo "⚙️ Running initial setup..."
"$LAUNCHER_PATH" setup

# Display success message
echo ""
echo "=========================================================="
echo "✅ KwaaiNet for Mac has been successfully installed!"
echo ""
echo "🚀 To start KwaaiNet, simply run:"
echo "   kwaainet start"
echo ""
echo "🔧 For custom options:"
echo "   kwaainet start --model \"unsloth/Llama-3.1-8B-Instruct\" --blocks 2 --port 8080"
echo ""
echo "📊 To view your configuration:"
echo "   kwaainet config --view"
echo ""
echo "📚 For more information, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"
echo "=========================================================="

# Notify about shell restart
echo ""
echo "Note: You may need to restart your terminal or run 'source ~/.$(basename $SHELL)rc'"
echo "for the 'kwaainet' command to be available in your PATH."
echo ""