#!/bin/bash

# KwaaiNet for Mac - One-Step Installer v0.2.14
# This script handles the entire installation process for KwaaiNet on macOS

set -e  # Exit on error

# Installer version
INSTALLER_VERSION="0.3.1"

# Set up logging
LOG_FILE="$HOME/kwaainet_install_$(date +%Y%m%d_%H%M%S).log"
# Log to file without duplicating terminal output
exec 3>&1 4>&2
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== KwaaiNet macOS Installer v$INSTALLER_VERSION ==="
echo "Installation started at: $(date)"
echo "Log file: $LOG_FILE"
echo "System: $(uname -a)"
echo ""
echo "This installer will set up KwaaiNet for sharing compute on macOS"
echo "It includes Python setup, dependencies, and environment configuration"
echo ""

# Parse command line arguments
SKIP_SYSTEM_PACKAGES=false
FORCE_CONDA=false
FORCE_VENV=false
NO_BUILD_TOOLS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-system-packages)
            SKIP_SYSTEM_PACKAGES=true
            shift
            ;;
        --force-conda)
            FORCE_CONDA=true
            shift
            ;;
        --force-venv)
            FORCE_VENV=true
            shift
            ;;
        --no-build-tools)
            NO_BUILD_TOOLS=true
            shift
            ;;
        --with-build-tools)
            NO_BUILD_TOOLS=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-system-packages  Skip system package installation (assumes all dependencies are available)"
            echo "  --force-conda         Force using conda environment instead of auto-detection"
            echo "  --force-venv          Force using virtual environment instead of auto-detection"
            echo "  --no-build-tools      Use pre-built wheels only (default - saves disk space)"
            echo "  --with-build-tools    Install build tools for compiling from source (requires extra disk space)"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Skip system packages: $SKIP_SYSTEM_PACKAGES"
echo "  Force conda: $FORCE_CONDA"
echo "  Force venv: $FORCE_VENV"
echo "  No build tools: $NO_BUILD_TOOLS"
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

    # Monitor space after Miniconda installation
    if command -v monitor_installation_space >/dev/null 2>&1; then
        monitor_installation_space "$HOME" "After Miniconda installation"
    fi
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

# Configure conda channels to avoid Terms of Service issues
echo "🔧 Configuring conda channels..."
# First try to accept TOS for existing channels if possible
echo "📝 Attempting to accept conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# If TOS acceptance fails, remove problematic channels and configure conda-forge only
if ! conda create -n temp_test_env python=3.10 -y --dry-run 2>/dev/null; then
    echo "⚠️ TOS acceptance failed, configuring conda-forge as exclusive channel..."
    conda config --remove channels https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda config --remove channels https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda config --remove channels defaults 2>/dev/null || true
    # Add conda-forge as primary and only channel
    conda config --add channels conda-forge
    conda config --set channel_priority strict
fi
echo "✅ Configured conda channels (avoids Terms of Service issues)"

# Source storage check and error diagnosis functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../common/storage_check.sh" ]; then
    source "$SCRIPT_DIR/../common/storage_check.sh"
else
    echo "⚠️ Warning: Storage check functions not found - continuing without space verification"
fi

if [ -f "$SCRIPT_DIR/../common/error_diagnosis.sh" ]; then
    source "$SCRIPT_DIR/../common/error_diagnosis.sh"
else
    echo "⚠️ Warning: Error diagnosis functions not found - using basic error handling"
fi

# Check storage requirements before installation
echo ""
echo "🔍 Verifying storage space requirements..."
SKIP_STORAGE_CHECK="${SKIP_STORAGE_CHECK:-false}"
if [ "$SKIP_STORAGE_CHECK" != "true" ] && command -v check_storage_requirements >/dev/null 2>&1; then
    # Determine installation parameters for space estimation
    USE_CONDA_FOR_SPACE="true"
    INSTALL_BUILD_TOOLS="true"
    if [ "$NO_BUILD_TOOLS" = "true" ]; then
        INSTALL_BUILD_TOOLS="false"
    fi

    # Check storage and exit if insufficient
    if ! check_storage_requirements "$HOME" "$USE_CONDA_FOR_SPACE" "$INSTALL_BUILD_TOOLS"; then
        echo ""
        echo "❌ Installation cannot proceed due to insufficient storage space."
        echo ""
        echo "💡 Options to continue:"
        echo "   1. Free up space using the suggestions above"
        echo "   2. Use space-saving options:"
        echo "      --no-build-tools     (saves ~1GB)"
        echo "   3. Use 'brew cleanup' to remove old Homebrew packages"
        echo "   4. Empty Trash and Downloads folder"
        echo "   5. Skip storage check: SKIP_STORAGE_CHECK=true bash installer.sh"
        echo ""
        exit 1
    fi

    # Monitor space during installation
    monitor_installation_space "$HOME" "Pre-installation"
else
    echo "ℹ️ Storage space check skipped"
fi
echo ""

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

# Uninstall any existing kwaainet packages to avoid version conflicts
echo "🧹 Removing any existing KwaaiNet packages..."
pip uninstall -y kwaainet-mac kwaainet_mac &>/dev/null || true

# Download and install the current project code
echo "📦 Downloading KwaaiNet source code..."

# Create permanent installation directory
INSTALL_DIR="$HOME/.kwaainet/source"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Remove any existing installation
rm -rf OpenAI-Petal* 2>/dev/null || true

if command -v git >/dev/null 2>&1; then
    echo "📡 Cloning repository with git..."
    git clone --depth 1 https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
    PROJECT_PATH="$INSTALL_DIR/OpenAI-Petal"
else
    echo "📡 Downloading repository archive..."
    curl -L https://github.com/Kwaai-AI-Lab/OpenAI-Petal/archive/main.tar.gz -o main.tar.gz
    tar -xzf main.tar.gz
    PROJECT_PATH="$INSTALL_DIR/OpenAI-Petal-main"
fi

# Install the current project in development mode from permanent location
echo "📦 Installing KwaaiNet for Mac in development mode..."
cd "$PROJECT_PATH/Installer/macOS"

# Add --only-binary flag if no build tools
# Use conda environment's pip explicitly to avoid broken system pip
CONDA_ENV_PIP="$CONDA_PREFIX/bin/pip"
if [ -x "$CONDA_ENV_PIP" ]; then
    PIP_INSTALL_CMD="$CONDA_ENV_PIP install -e ."
    if [ "$NO_BUILD_TOOLS" = true ]; then
        PIP_INSTALL_CMD="$CONDA_ENV_PIP install -e . --only-binary=all"
        echo "ℹ️ Using pre-built wheels only (--only-binary=all) to avoid compilation"
    fi
else
    # Fallback to regular pip if conda env pip not found
    PIP_INSTALL_CMD="pip install -e ."
    if [ "$NO_BUILD_TOOLS" = true ]; then
        PIP_INSTALL_CMD="pip install -e . --only-binary=all"
        echo "ℹ️ Using pre-built wheels only (--only-binary=all) to avoid compilation"
    fi
fi

# Use monitored installation if available
if command -v monitor_package_installation >/dev/null 2>&1; then
    if ! monitor_package_installation "KwaaiNet macOS package" "$PIP_INSTALL_CMD" "$HOME"; then
        echo "❌ Failed to install KwaaiNet package"
        exit 1
    fi
else
    # Fallback to standard installation
    if $PIP_INSTALL_CMD; then
        echo "✅ KwaaiNet macOS package installed successfully"
    else
        echo "❌ Failed to install KwaaiNet package"
        if command -v diagnose_pip_failure >/dev/null 2>&1; then
            diagnose_pip_failure $? "KwaaiNet package installation failed" "kwaainet" "$HOME"
        else
            echo "Please check your internet connection and available disk space."
        fi
        exit 1
    fi
fi

# Run initial setup now that the package is installed and environment is ready
echo "⚙️ Running initial setup..."
python -c "import kwaainet.installer; kwaainet.installer.setup_mac()"

# Monitor space after main package installation
if command -v monitor_installation_space >/dev/null 2>&1; then
    monitor_installation_space "$HOME" "After KwaaiNet package installation"
fi

# Create launcher script for one-step execution
echo "🚀 Creating launcher script..."
LAUNCHER_PATH="$HOME/.local/bin/kwaainet"
mkdir -p "$HOME/.local/bin"

cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first

# Find conda installation (prioritize user installations over system)
CONDA_PATH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/usr/local/Caskroom/miniconda/base"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda"
elif command -v conda >/dev/null 2>&1; then
    # Try to get conda base, but verify it has conda.sh
    POTENTIAL_PATH=$(conda info --base 2>/dev/null)
    if [ -n "$POTENTIAL_PATH" ] && [ -f "$POTENTIAL_PATH/etc/profile.d/conda.sh" ]; then
        CONDA_PATH="$POTENTIAL_PATH"
    else
        # Fallback to directory detection
        CONDA_PATH=$(dirname $(dirname $(which conda)) 2>/dev/null)
    fi
fi

if [ -z "$CONDA_PATH" ] || [ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    echo "❌ Error: Could not find conda installation with conda.sh script."
    echo "Expected locations:"
    echo "  - $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "  - $HOME/anaconda3/etc/profile.d/conda.sh"
    echo "  - /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    echo "  - /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    exit 1
fi

# Source conda without changing the prompt
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the environment and run the command
conda activate kwaainet

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate kwaainet conda environment."
    exit 1
fi

# Run kwaainet command directly via Python to avoid module import recursion
if [ "$1" = "setup" ]; then
    exec python -c "import kwaainet.installer; kwaainet.installer.setup_mac()"
else
    exec python -c "
import sys
sys.argv = ['kwaainet'] + sys.argv[1:]
import kwaainet.runner
kwaainet.runner.main()
" "$@"
fi
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
# Initial setup is now handled after package installation is complete

# Final space monitoring
if command -v monitor_installation_space >/dev/null 2>&1; then
    monitor_installation_space "$HOME" "Installation completed"
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