#!/bin/bash

# KwaaiNet for Mac - One-Step Installer v0.1
# This script handles the entire installation process for KwaaiNet on macOS

set -e  # Exit on error

# Installer version
INSTALLER_VERSION="0.1.3"

echo "=========================================================="
echo "KwaaiNet for Mac - One-Step Installer v$INSTALLER_VERSION"
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
    
    # Install via Homebrew
    brew install --cask miniconda
    
    # Initialize conda for shell
    if [[ "$(uname -m)" == "arm64" ]]; then
        CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
    else
        CONDA_PATH="/usr/local/Caskroom/miniconda/base"
    fi
    
    $CONDA_PATH/bin/conda init "$(basename "${SHELL}")"
    
    echo "✅ Miniconda installed successfully"
}

# Function to ensure shell configuration file exists
ensure_shell_config() {
    local shell_name="$(basename "${SHELL}")"
    local rc_file="$HOME/.${shell_name}rc"
    
    # Create the rc file if it doesn't exist
    if [[ ! -f "$rc_file" ]]; then
        echo "Creating shell config file: $rc_file"
        touch "$rc_file"
    fi
    
    echo "$rc_file"
}

# Function to check if a line exists in a file and add it if not
add_line_if_not_exists() {
    local file="$1"
    local line="$2"
    local comment="$3"
    
    # Escape the line for grep
    local escaped_line=$(echo "$line" | sed 's/[]\/$*.^|[]/\\&/g')
    
    if ! grep -q "$escaped_line" "$file"; then
        if [ -n "$comment" ]; then
            echo "" >> "$file"
            echo "$comment" >> "$file"
        fi
        echo "$line" >> "$file"
        return 0
    fi
    return 1
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
    if [[ "$(uname -m)" == "arm64" ]]; then
        CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
    else
        CONDA_PATH="/usr/local/Caskroom/miniconda/base"
    fi
    
    if [[ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    fi
else
    echo "✅ Conda already installed"
    
    # Try to locate conda installation
    if command -v conda >/dev/null 2>&1; then
        CONDA_PATH=$(dirname $(dirname $(which conda)))
        if [[ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]]; then
            . "$CONDA_PATH/etc/profile.d/conda.sh"
        fi
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
# First make sure conda is initialized for this session
if command_exists conda; then
    echo "🔄 Initializing conda for current session..."
    # Get conda path
    CONDA_EXEC=$(which conda)
    CONDA_PATH=$(dirname $(dirname $CONDA_EXEC))
    
    # Source conda.sh to allow conda activate in the current shell
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        . "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        . "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    else
        echo "⚠️ Could not find conda.sh, trying alternative activation method..."
    fi
fi

# Now try to activate
if ! conda activate kwaainet 2>/dev/null; then
    if ! source activate kwaainet 2>/dev/null; then
        echo "⚠️ Could not activate conda environment. Using direct Python path..."
        # Try direct approach with explicit path
        if [[ "$(uname -m)" == "arm64" ]]; then
            PYTHON_PATH="/opt/homebrew/Caskroom/miniconda/base/envs/kwaainet/bin/python"
        else
            PYTHON_PATH="/usr/local/Caskroom/miniconda/base/envs/kwaainet/bin/python"
        fi
        
        if [ -f "$PYTHON_PATH" ]; then
            # Use this Python directly for the pip install
            echo "✅ Found Python at $PYTHON_PATH"
            alias python="$PYTHON_PATH"
        else
            echo "❌ Error: Could not find Python in the kwaainet environment."
            echo "Please manually run: conda activate kwaainet"
            exit 1
        fi
    else
        echo "✅ Environment activated using source activate"
    fi
else
    echo "✅ Environment activated using conda activate"
fi

# Clear cached versions of the package
echo "🧹 Clearing any cached versions of KwaaiNet..."
pip cache remove kwaainet-mac &>/dev/null || true
pip cache remove kwaainet_mac &>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Install the package directly from your URL
echo "📦 Installing KwaaiNet for Mac..."
pip install --no-cache-dir https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.8.0.tar.gz

# Create launcher script for one-step execution
echo "🚀 Creating launcher script..."
LAUNCHER_PATH="$HOME/.local/bin/kwaainet"
mkdir -p "$HOME/.local/bin"

cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first

# Find conda installation
if command -v conda >/dev/null 2>&1; then
    CONDA_PATH=$(dirname $(dirname $(which conda)))
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/usr/local/Caskroom/miniconda/base"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
else
    echo "❌ Error: Could not find conda installation."
    exit 1
fi

# Source conda without changing the prompt
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "❌ Error: Could not find conda.sh in $CONDA_PATH"
    exit 1
fi

# Activate the environment and run the command
conda activate kwaainet && python -m kwaainet.runner "$@"
EOF

chmod +x "$LAUNCHER_PATH"

# Get the shell configuration file path
SHELL_RC=$(ensure_shell_config)

# Add to PATH if not already there
echo "📝 Ensuring ~/.local/bin is in PATH..."

# Check and update common shell configuration files
SHELL_FILES=(
    "$HOME/.zshrc"
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.profile"
)

PATH_UPDATED=false
for rc_file in "${SHELL_FILES[@]}"; do
    if [ -f "$rc_file" ]; then
        if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$rc_file"; then
            echo "📝 Adding ~/.local/bin to PATH in $rc_file"
            echo "" >> "$rc_file"
            echo "# Added by KwaaiNet installer" >> "$rc_file"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
            PATH_UPDATED=true
        else
            echo "✅ PATH already configured in $rc_file"
        fi
    fi
done

# Also update conda's activate.d to ensure PATH is set when conda is activated
if command_exists conda; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        CONDA_ACTIVATE_DIR="$CONDA_BASE/etc/conda/activate.d"
        mkdir -p "$CONDA_ACTIVATE_DIR"
        if [ ! -f "$CONDA_ACTIVATE_DIR/kwaainet_path.sh" ]; then
            echo "📝 Adding PATH configuration for conda environments"
            echo '#!/bin/bash' > "$CONDA_ACTIVATE_DIR/kwaainet_path.sh"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CONDA_ACTIVATE_DIR/kwaainet_path.sh"
            chmod +x "$CONDA_ACTIVATE_DIR/kwaainet_path.sh"
            PATH_UPDATED=true
        fi
    fi
fi

# If path wasn't in any existing file, create/update the default for current shell
if [ "$PATH_UPDATED" = false ]; then
    SHELL_RC=$(ensure_shell_config)
    echo "📝 Adding ~/.local/bin to PATH in $SHELL_RC (default)"
    echo "" >> "$SHELL_RC"
    echo "# Added by KwaaiNet installer" >> "$SHELL_RC"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
fi

# Update current session PATH
export PATH="$HOME/.local/bin:$PATH"

# Ensure conda initialization in shell config if needed
if ! grep -q "conda initialize" "$SHELL_RC"; then
    echo "📝 Adding conda initialization to $SHELL_RC"
    # Find conda path first
    if [[ "$(uname -m)" == "arm64" ]]; then
        CONDA_FULL_PATH="/opt/homebrew/Caskroom/miniconda/base"
    else
        CONDA_FULL_PATH="/usr/local/Caskroom/miniconda/base"
    fi
    
    if [ -f "$CONDA_FULL_PATH/bin/conda" ]; then
        echo "✅ Found conda at $CONDA_FULL_PATH"
        # Run conda init and capture its output
        "$CONDA_FULL_PATH/bin/conda" init "$(basename "${SHELL}")" > /dev/null
        echo "✅ Added conda initialization to $SHELL_RC"
    else
        # Try alternative approach if path is not as expected
        if command_exists conda; then
            echo "✅ Using existing conda command"
            conda init "$(basename "${SHELL}")" > /dev/null
            echo "✅ Added conda initialization to $SHELL_RC"
        else
            echo "⚠️ Could not automatically add conda initialization to $SHELL_RC"
            echo "⚠️ You may need to run 'conda init' manually after installation"
        fi
    fi
else
    echo "✅ Conda initialization already in $SHELL_RC"
fi

# Try to create system-wide symlink if possible
if [ -w "/usr/local/bin" ]; then
    echo "📌 Creating system-wide link in /usr/local/bin..."
    ln -sf "$LAUNCHER_PATH" /usr/local/bin/kwaainet
fi

# Run initial setup
echo "⚙️ Running initial setup..."
# Use the direct PATH to kwaainet if available, otherwise use the launcher script
if command_exists conda; then
    # Try to activate conda and run directly
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
        if conda activate kwaainet 2>/dev/null; then
            python -m kwaainet.runner setup
        else
            "$LAUNCHER_PATH" setup
        fi
    else
        "$LAUNCHER_PATH" setup
    fi
else
    "$LAUNCHER_PATH" setup
fi

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
echo "Note: You may need to restart your terminal or run 'source \"$SHELL_RC\"'"
echo "for the 'kwaainet' command to be available in your PATH."
echo ""