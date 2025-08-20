#!/bin/bash

# KwaaiNet for Linux - One-Step Installer
# This script handles the entire installation process for KwaaiNet on Linux

set -e  # Exit on error

echo "=========================================================="
echo "KwaaiNet for Linux - One-Step Installer"
echo "=========================================================="
echo "This installer will set up KwaaiNet for sharing compute on Linux"
echo "It includes Python setup, dependencies, and environment configuration"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to show a spinner while a command runs
show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    echo -n " "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        DISTRO_FAMILY=""
        
        case $DISTRO in
            ubuntu|debian|linuxmint|pop|elementary)
                DISTRO_FAMILY="debian"
                PKG_MANAGER="apt"
                PKG_UPDATE="apt update"
                PKG_INSTALL="apt install -y"
                ;;
            fedora|centos|rhel|rocky|almalinux)
                DISTRO_FAMILY="redhat"
                if command_exists dnf; then
                    PKG_MANAGER="dnf"
                    PKG_UPDATE="dnf check-update || true"
                    PKG_INSTALL="dnf install -y"
                else
                    PKG_MANAGER="yum"
                    PKG_UPDATE="yum check-update || true"
                    PKG_INSTALL="yum install -y"
                fi
                ;;
            arch|manjaro|endeavouros)
                DISTRO_FAMILY="arch"
                PKG_MANAGER="pacman"
                PKG_UPDATE="pacman -Sy"
                PKG_INSTALL="pacman -S --noconfirm"
                ;;
            opensuse*|sles)
                DISTRO_FAMILY="suse"
                PKG_MANAGER="zypper"
                PKG_UPDATE="zypper refresh"
                PKG_INSTALL="zypper install -y"
                ;;
            *)
                DISTRO_FAMILY="unknown"
                echo "⚠️ Unknown distribution: $DISTRO"
                echo "Attempting to use generic commands..."
                ;;
        esac
        
        echo "✅ Detected: $DISTRO $DISTRO_VERSION ($DISTRO_FAMILY family)"
        echo "📦 Package manager: $PKG_MANAGER"
    else
        echo "❌ Error: Cannot detect Linux distribution"
        exit 1
    fi
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        echo "⚠️ Warning: Running as root. This installer should be run as a regular user."
        echo "Some operations will use sudo when needed."
        USE_SUDO=""
    else
        USE_SUDO="sudo"
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo "📦 Installing system dependencies..."
    
    case $DISTRO_FAMILY in
        debian)
            $USE_SUDO $PKG_UPDATE
            $USE_SUDO $PKG_INSTALL curl wget git build-essential python3 python3-pip python3-venv python3-dev pciutils
            # GPU support packages
            $USE_SUDO $PKG_INSTALL mesa-utils || true
            ;;
        redhat)
            $USE_SUDO $PKG_UPDATE
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages
            $USE_SUDO $PKG_INSTALL mesa-dri-drivers || true
            ;;
        arch)
            $USE_SUDO $PKG_UPDATE
            $USE_SUDO $PKG_INSTALL curl wget git base-devel python python-pip pciutils
            # GPU support packages
            $USE_SUDO $PKG_INSTALL mesa || true
            ;;
        suse)
            $USE_SUDO $PKG_UPDATE
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages
            $USE_SUDO $PKG_INSTALL Mesa || true
            ;;
        *)
            echo "⚠️ Unknown distribution family. Please install the following manually:"
            echo "  - curl, wget, git"
            echo "  - Python 3.8+ with pip and development headers"
            echo "  - Build tools (gcc, make)"
            ;;
    esac
    
    echo "✅ System dependencies installed"
}

# Function to detect GPU
detect_gpu() {
    echo "🔍 Detecting GPU hardware..."
    
    GPU_TYPE="none"
    GPU_INFO=""
    
    # Check if lspci is available
    if ! command_exists lspci; then
        echo "⚠️ lspci command not available. Limited GPU detection."
        return 0
    fi
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "✅ NVIDIA GPU detected: $GPU_INFO"
    elif lspci 2>/dev/null | grep -i nvidia >/dev/null 2>&1; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(lspci 2>/dev/null | grep -i nvidia | head -1)
        echo "✅ NVIDIA GPU detected: $GPU_INFO"
        echo "⚠️ NVIDIA drivers may not be installed. GPU acceleration might not work."
    # Check for AMD GPU
    elif command_exists rocm-smi; then
        GPU_TYPE="amd"
        GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1)
        echo "✅ AMD GPU detected: $GPU_INFO"
    elif lspci 2>/dev/null | grep -i amd | grep -i vga >/dev/null 2>&1; then
        GPU_TYPE="amd"
        GPU_INFO=$(lspci 2>/dev/null | grep -i amd | grep -i vga | head -1)
        echo "✅ AMD GPU detected: $GPU_INFO"
        echo "⚠️ ROCm may not be installed. GPU acceleration might not work."
    # Check for Intel GPU
    elif lspci 2>/dev/null | grep -i intel | grep -i vga >/dev/null 2>&1; then
        GPU_TYPE="intel"
        GPU_INFO=$(lspci 2>/dev/null | grep -i intel | grep -i vga | head -1)
        echo "✅ Intel GPU detected: $GPU_INFO"
    else
        echo "ℹ️ No dedicated GPU detected. Using CPU-only mode."
    fi
    
    export GPU_TYPE GPU_INFO
}

# Function to install Miniconda
install_miniconda() {
    echo "🐍 Installing Miniconda..."
    
    # Check if conda is already installed
    if command_exists conda; then
        echo "✅ Conda already installed"
        return 0
    fi
    
    # Determine architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            CONDA_ARCH="x86_64"
            ;;
        aarch64|arm64)
            CONDA_ARCH="aarch64"
            ;;
        *)
            echo "❌ Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Download and install Miniconda
    CONDA_INSTALLER="Miniconda3-latest-Linux-${CONDA_ARCH}.sh"
    CONDA_URL="https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}"
    
    echo "📥 Downloading Miniconda..."
    wget -q "$CONDA_URL" -O "/tmp/$CONDA_INSTALLER"
    
    echo "🔧 Installing Miniconda..."
    bash "/tmp/$CONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Clean up
    rm "/tmp/$CONDA_INSTALLER"
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null || true
    
    # Source conda for current session
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    echo "✅ Miniconda installed successfully"
}

# Function to choose Python environment method
choose_python_method() {
    echo "🐍 Choosing Python environment method..."
    
    PYTHON_METHOD=""
    
    # Check if user prefers system Python or conda
    if [ "${KWAAINET_USE_SYSTEM_PYTHON:-}" = "true" ]; then
        PYTHON_METHOD="system"
        echo "ℹ️ Using system Python (KWAAINET_USE_SYSTEM_PYTHON=true)"
    elif [ "${KWAAINET_USE_CONDA:-}" = "true" ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda (KWAAINET_USE_CONDA=true)"
    else
        # Auto-detect best method
        if command_exists conda; then
            PYTHON_METHOD="conda"
            echo "✅ Using existing conda installation"
        elif python3 --version >/dev/null 2>&1; then
            PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            # Use simple version comparison instead of bc
            MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
            MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
            if [ "$MAJOR" -gt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -ge 8 ]; then
                PYTHON_METHOD="system"
                echo "✅ Using system Python $PYTHON_VERSION"
            else
                echo "⚠️ System Python is too old ($PYTHON_VERSION). Installing conda..."
                PYTHON_METHOD="conda"
            fi
        else
            echo "⚠️ No suitable Python found. Installing conda..."
            PYTHON_METHOD="conda"
        fi
    fi
    
    export PYTHON_METHOD
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

# Main installation flow starts here
echo "🔍 Detecting system configuration..."

# Detect distribution
detect_distro

# Check root privileges
check_root

# Detect GPU
detect_gpu

# Install system dependencies
install_system_deps

# Choose Python method
choose_python_method

# Install Python environment if needed
if [ "$PYTHON_METHOD" = "conda" ]; then
    if ! command_exists conda; then
        install_miniconda
        # Re-source conda after installation
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
    fi
    
    # Create environment
    echo "⚙️ Setting up KwaaiNet conda environment..."
    if ! conda info --envs | grep -q kwaainet; then
        conda create -y -n kwaainet python=3.10
        echo "✅ Created Python 3.10 environment for KwaaiNet"
    else
        echo "✅ Using existing kwaainet environment"
    fi
    
    # Activate environment (ensure conda is initialized first)
    if ! conda activate kwaainet 2>/dev/null; then
        echo "⚠️ Conda not initialized properly. Initializing conda..."
        conda init bash
        conda init zsh 2>/dev/null || true
        # Source conda for current session
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
        conda activate kwaainet
    fi
    PYTHON_EXEC="python"
    PIP_EXEC="pip"
    
elif [ "$PYTHON_METHOD" = "system" ]; then
    # Use system Python with virtual environment
    echo "⚙️ Setting up KwaaiNet virtual environment..."
    
    VENV_PATH="$HOME/.kwaainet-venv"
    if [ ! -d "$VENV_PATH" ]; then
        python3 -m venv "$VENV_PATH"
        echo "✅ Created virtual environment for KwaaiNet"
    else
        echo "✅ Using existing virtual environment"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    PYTHON_EXEC="$VENV_PATH/bin/python"
    PIP_EXEC="$VENV_PATH/bin/pip"
fi

# Clear cached versions of the package
echo "🧹 Clearing any cached versions of KwaaiNet..."
$PIP_EXEC cache remove kwaainet-linux &>/dev/null || true
$PIP_EXEC cache remove kwaainet_linux &>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Install the package
echo "📦 Installing KwaaiNet for Linux..."

# Install basic dependencies first
$PIP_EXEC install pyyaml &>/dev/null || {
    echo "⚠️ Failed to install pyyaml. Continuing..."
}

# Install updated petals with rope_scaling support
echo "📦 Installing Petals 2.3.0.dev2 with rope_scaling support..."
echo "   This may take several minutes as it builds from source..."

# Try to install with progress bar, fallback to verbose if progress bar not supported
if $PIP_EXEC install git+https://github.com/bigscience-workshop/petals.git 2>/dev/null; then
    echo "✅ Petals installed successfully from git"
elif $PIP_EXEC install git+https://github.com/bigscience-workshop/petals.git -v 2>/dev/null; then
    echo "✅ Petals installed successfully from git (verbose mode)"
else
    echo "⚠️ Failed to install petals from git. Trying fallback installation..."
    if $PIP_EXEC install petals 2>/dev/null || $PIP_EXEC install petals -v; then
        echo "✅ Petals installed from PyPI"
    else
        echo "⚠️ Failed to install petals. Continuing with local installation..."
    fi
fi

# Upgrade transformers and huggingface_hub for compatibility
echo "📦 Upgrading transformers and huggingface_hub for Llama 3.1 rope_scaling support..."
if $PIP_EXEC install --upgrade "transformers>=4.43.1" "huggingface_hub>=0.20.0"; then
    echo "✅ Successfully upgraded transformers and huggingface_hub"
else
    echo "⚠️ Failed to upgrade transformers/huggingface_hub. May have compatibility issues with Llama 3.1 models..."
fi

# Install from the local development version
INSTALLER_DIR="$(dirname "$0")"
if [ -d "$INSTALLER_DIR/linux" ]; then
    echo "📦 Installing from local development version..."
    if ! $PIP_EXEC install -e "$INSTALLER_DIR/linux/" 2>/dev/null; then
        echo "⚠️ Failed to install local development version. Installing dependencies only..."
        echo "📦 Installing PyTorch (CPU version)..."
        echo "   This may take a few minutes to download..."
        if $PIP_EXEC install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            echo "✅ PyTorch installed successfully"
        else
            echo "❌ Failed to install PyTorch. Please check your internet connection."
            exit 1
        fi
    fi
else
    echo "⚠️ Local development version not found. Please build the Linux package first."
    echo "📦 Installing PyTorch dependencies..."
    echo "   This may take a few minutes to download..."
    if $PIP_EXEC install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
        echo "✅ PyTorch installed successfully"
    else
        echo "❌ Failed to install PyTorch. Please check your internet connection."
        exit 1
    fi
fi

# Create launcher script for one-step execution
echo "🚀 Creating launcher script..."
LAUNCHER_PATH="$HOME/.local/bin/kwaainet"
mkdir -p "$HOME/.local/bin"

if [ "$PYTHON_METHOD" = "conda" ]; then
    # Conda-based launcher
    cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first

# Find conda installation
if command -v conda >/dev/null 2>&1; then
    CONDA_PATH=$(dirname $(dirname $(which conda)))
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
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

else
    # Virtual environment launcher
    VENV_PATH="$HOME/.kwaainet-venv"
    cat > "$LAUNCHER_PATH" << EOF
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet from virtual environment

VENV_PATH="$VENV_PATH"

if [ ! -d "\$VENV_PATH" ]; then
    echo "❌ Error: KwaaiNet virtual environment not found at \$VENV_PATH"
    exit 1
fi

# Activate virtual environment and run the command
source "\$VENV_PATH/bin/activate" && python -m kwaainet.runner "\$@"
EOF
fi

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
            add_line_if_not_exists "$rc_file" 'export PATH="$HOME/.local/bin:$PATH"' "# Added by KwaaiNet installer"
            PATH_UPDATED=true
        else
            echo "✅ PATH already configured in $rc_file"
        fi
    fi
done

# If path wasn't in any existing file, create/update the default for current shell
if [ "$PATH_UPDATED" = false ]; then
    SHELL_RC=$(ensure_shell_config)
    echo "📝 Adding ~/.local/bin to PATH in $SHELL_RC (default)"
    add_line_if_not_exists "$SHELL_RC" 'export PATH="$HOME/.local/bin:$PATH"' "# Added by KwaaiNet installer"
fi

# Update current session PATH
export PATH="$HOME/.local/bin:$PATH"

# Try to create system-wide symlink if possible
if [ -w "/usr/local/bin" ]; then
    echo "📌 Creating system-wide link in /usr/local/bin..."
    ln -sf "$LAUNCHER_PATH" /usr/local/bin/kwaainet
fi

# Run initial setup
echo "⚙️ Running initial setup..."
if [ "$PYTHON_METHOD" = "conda" ]; then
    if conda activate kwaainet 2>/dev/null; then
        python -m kwaainet.runner setup 2>/dev/null || {
            echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
        }
    else
        "$LAUNCHER_PATH" setup 2>/dev/null || {
            echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
        }
    fi
else
    "$LAUNCHER_PATH" setup 2>/dev/null || {
        echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
    }
fi

echo ""
echo "=========================================================="
echo "✅ KwaaiNet for Linux installation in progress!"
echo ""
echo "🔧 Configuration detected:"
echo "   - Distribution: $DISTRO $DISTRO_VERSION"
echo "   - GPU: $GPU_TYPE $([ -n "$GPU_INFO" ] && echo "($GPU_INFO)" || echo "")"
echo "   - Python method: $PYTHON_METHOD"
echo ""
echo "📝 Next steps:"
echo "   1. Complete the Python package installation"
echo "   2. Set up launcher scripts"
echo "   3. Configure GPU acceleration (if available)"
echo ""
echo "📚 For more information, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"
echo "=========================================================="