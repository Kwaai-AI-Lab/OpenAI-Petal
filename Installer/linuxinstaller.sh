#!/bin/bash

# KwaaiNet for Linux - One-Step Installer v0.1
# This script handles the entire installation process for KwaaiNet on Linux

set -e  # Exit on error

# Installer version
INSTALLER_VERSION="0.2.0"

# Parse command line arguments
SKIP_SYSTEM_PACKAGES=false
FORCE_CONDA=false
FORCE_VENV=false
NO_BUILD_TOOLS=false

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-system-packages  Skip system package installation (assumes all dependencies are available)"
            echo "  --force-conda         Force using conda environment instead of auto-detection"
            echo "  --force-venv          Force using virtual environment instead of auto-detection"
            echo "  --no-build-tools      Skip packages requiring build tools, use pre-built wheels only"
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

echo "=========================================================="
echo "KwaaiNet for Linux - One-Step Installer v$INSTALLER_VERSION"
echo "=========================================================="
echo "This installer will set up KwaaiNet for sharing compute on Linux"
echo "It includes Python setup, dependencies, and environment configuration"
if [ "$SKIP_SYSTEM_PACKAGES" = true ]; then
    echo "⚠️ Skipping system package installation (--no-system-packages)"
fi
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get the appropriate pip command for the current environment
get_pip_command() {
    # If we're in a conda environment, use pip directly
    if [ "${CONDA_DEFAULT_ENV:-}" = "kwaainet" ] || [ "${PYTHON_METHOD:-}" = "conda" ]; then
        echo "pip"
    # If we're in a venv and it has its own pip, use it
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/pip" ]; then
        echo "${VIRTUAL_ENV}/bin/pip"
    # Otherwise, try to find the best system pip
    elif command_exists pip3; then
        echo "pip3"
    elif command_exists pip && ${PYTHON_CMD:-python3} -c "import sys; exit(0 if sys.version_info[0] == 3 else 1)" 2>/dev/null; then
        echo "pip"
    else
        echo "${PYTHON_CMD:-python3} -m pip"
    fi
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

# Function to check if system dependencies are available
check_system_deps() {
    echo "🔍 Checking system dependencies..."
    
    local missing_essential=()
    local missing_optional=()
    local missing_build=()
    
    # Check essential commands
    if ! command_exists curl; then
        missing_essential+=("curl")
    fi
    
    if ! command_exists wget; then
        missing_essential+=("wget") 
    fi
    
    if ! command_exists git; then
        missing_essential+=("git")
    fi
    
    # Find best Python version
    PYTHON_CMD=""
    PYTHON_VERSION=""
    
    # Try to find the newest Python version (check 3.12 down to 3.7)
    for ver in 3.12 3.11 3.10 3.9 3.8 3.7; do
        if command_exists "python$ver"; then
            PYTHON_CMD="python$ver"
            PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
            echo "✅ Found Python $PYTHON_VERSION at $PYTHON_CMD"
            break
        fi
    done
    
    # Fallback to python3 if specific versions not found
    if [ -z "$PYTHON_CMD" ] && command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
        echo "✅ Found Python $PYTHON_VERSION at python3"
    fi
    
    if [ -z "$PYTHON_CMD" ]; then
        missing_essential+=("python3")
    else
        # Check Python version
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
            echo "⚠️ Python $PYTHON_VERSION found, but Python 3.7+ is required"
            missing_essential+=("python3 (3.7+)")
        elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 7 ]; then
            echo "⚠️ Python 3.7 detected. This may not be compatible with all dependencies."
            echo "   Modern ML libraries (transformers, pytorch) typically require Python 3.8+."
            echo "   The installer will try to continue but may fail during package installation."
            echo ""
            read -p "   Continue anyway? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Installation cancelled. Please upgrade to Python 3.8+ for best compatibility."
                exit 1
            fi
        fi
        
        # Export the best Python command for later use
        export PYTHON_CMD
    fi
    
    # Check pip - any working pip interface is acceptable
    # On Ubuntu 24.04+, pip might only be available via 'python3 -m pip'
    local pip_available=false
    if command_exists pip3; then
        pip_available=true
    elif command_exists pip; then
        pip_available=true  
    elif ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        pip_available=true
    elif ${PYTHON_CMD:-python3} -c "import ensurepip" >/dev/null 2>&1; then
        # ensurepip is available, we can bootstrap pip
        pip_available=true
    fi
    
    if [ "$pip_available" = false ]; then
        missing_essential+=("pip (python package manager)")
    fi
    
    # Check build tools (can potentially be handled by conda or skipped)
    if [ "$NO_BUILD_TOOLS" = true ]; then
        echo "ℹ️ Skipping build tools check (--no-build-tools flag)"
    else
        if ! command_exists gcc && ! command_exists clang; then
            missing_build+=("build tools (gcc/clang)")
        fi
        
        # Check for Rust compiler (needed for tokenizers)
        if ! command_exists rustc; then
            missing_build+=("rust compiler (for tokenizers)")
        fi
    fi
    
    # Check development headers (needed for some Python packages)
    if ! ${PYTHON_CMD:-python3} -c "import sysconfig; import os; print(os.path.exists(sysconfig.get_path('include')))" 2>/dev/null | grep -q True; then
        missing_optional+=("python3-dev headers")
    fi
    
    # Set global variables for different dependency types
    export MISSING_ESSENTIAL=("${missing_essential[@]}")
    export MISSING_BUILD=("${missing_build[@]}")  
    export MISSING_OPTIONAL=("${missing_optional[@]}")
    
    if [ ${#missing_essential[@]} -eq 0 ]; then
        if [ ${#missing_build[@]} -gt 0 ] || [ ${#missing_optional[@]} -gt 0 ]; then
            echo "✅ Essential dependencies available"
            [ ${#missing_build[@]} -gt 0 ] && echo "⚠️ Missing build tools: ${missing_build[*]} (can be provided by conda)"
            [ ${#missing_optional[@]} -gt 0 ] && echo "⚠️ Missing optional: ${missing_optional[*]}"
            return 2  # Partial success - essential OK, build tools missing
        else
            echo "✅ All dependencies are available"
            return 0  # Full success
        fi
    else
        echo "❌ Missing essential dependencies: ${missing_essential[*]}"
        [ ${#missing_build[@]} -gt 0 ] && echo "❌ Missing build tools: ${missing_build[*]}"
        [ ${#missing_optional[@]} -gt 0 ] && echo "⚠️ Missing optional: ${missing_optional[*]}"
        return 1  # Failure
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo "📦 Installing system dependencies..."
    echo "🔍 Debug: Starting install_system_deps function"
    
    # Check what we need to install
    check_system_deps
    local dep_status=$?
    
    echo "🔍 Debug: Dependency check status: $dep_status"
    
    if [ $dep_status -eq 0 ]; then
        echo "✅ System dependencies already satisfied"
        return 0
    elif [ $dep_status -eq 2 ]; then
        echo "ℹ️ Essential dependencies satisfied, build tools missing"
        echo "ℹ️ Will use conda to provide build tools and avoid sudo"
        export FORCE_CONDA_FOR_BUILD_TOOLS=true
        return 0
    fi
    
    # Debug: Show what dependencies are detected as missing
    echo "🔍 Debug: Missing essential dependencies: ${MISSING_ESSENTIAL[*]:-none}"
    echo "🔍 Debug: Number of missing essential: ${#MISSING_ESSENTIAL[@]}"
    
    # Only need sudo if essential packages are missing
    if [ ${#MISSING_ESSENTIAL[@]} -gt 0 ]; then
        if ! command_exists sudo && [ "$EUID" -ne 0 ]; then
            echo "❌ Error: Essential system packages need to be installed but sudo is not available."
            echo "Please install the missing essential dependencies manually or run as root."
            echo "Required: ${MISSING_ESSENTIAL[*]}"
            echo ""
            echo "Alternatively, install these packages manually:"
            case $DISTRO_FAMILY in
                debian) echo "  sudo apt update && sudo apt install ${MISSING_ESSENTIAL[*]// / }" ;;
                redhat) echo "  sudo yum install ${MISSING_ESSENTIAL[*]// / }" ;;
                arch) echo "  sudo pacman -S ${MISSING_ESSENTIAL[*]// / }" ;;
                suse) echo "  sudo zypper install ${MISSING_ESSENTIAL[*]// / }" ;;
            esac
            exit 1
        fi
    else
        echo "ℹ️ Only build tools are missing - will use conda to provide them"
        export FORCE_CONDA_FOR_BUILD_TOOLS=true
        return 0
    fi
    
    case $DISTRO_FAMILY in
        debian)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git build-essential python3 python3-venv python3-dev pciutils
            
            # Handle pip installation for Ubuntu 24.04+
            echo "🔄 Setting up pip for Python package management..."
            if ! $USE_SUDO $PKG_INSTALL python3-pip 2>/dev/null; then
                echo "ℹ️ python3-pip package not available, using ensurepip approach..."
                # Try to install ensurepip package
                $USE_SUDO $PKG_INSTALL python3-ensurepip 2>/dev/null || true
            fi
            
            # Ensure pip is available via some method
            if ! command_exists pip3 && ! command_exists pip && ! ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                echo "🔄 Bootstrapping pip using ensurepip..."
                ${PYTHON_CMD:-python3} -m ensurepip --upgrade 2>/dev/null || true
                # If that fails, try without --upgrade
                if ! ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                    echo "🔄 Attempting basic ensurepip bootstrap..."
                    ${PYTHON_CMD:-python3} -m ensurepip 2>/dev/null || true
                fi
            fi
            
            # Final verification
            if ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                echo "✅ pip is available via '${PYTHON_CMD:-python3} -m pip'"
            elif command_exists pip3; then
                echo "✅ pip3 command is available"
            elif command_exists pip; then
                echo "✅ pip command is available"
            else
                echo "⚠️ pip installation may have failed, but continuing..."
            fi
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa-utils || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if command_exists snap && $USE_SUDO snap install rustup --classic 2>/dev/null; then
                    echo "✅ Rust installed via snap"
                    export PATH="$PATH:/snap/bin"
                    /snap/bin/rustup default stable 2>/dev/null || true
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        redhat)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa-dri-drivers || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                # Try to install rust via package manager first
                if $USE_SUDO $PKG_INSTALL rust cargo 2>/dev/null; then
                    echo "✅ Rust installed via package manager"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        arch)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git base-devel python python-pip pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if $USE_SUDO $PKG_INSTALL rust 2>/dev/null; then
                    echo "✅ Rust installed via pacman"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        suse)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL Mesa || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if $USE_SUDO $PKG_INSTALL rust cargo 2>/dev/null; then
                    echo "✅ Rust installed via zypper"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        *)
            echo "❌ Unknown distribution family. Please install the following manually:"
            echo "  - curl, wget, git"
            echo "  - Python 3.8+ with pip and development headers"
            echo "  - Build tools (gcc, make)"
            exit 1
            ;;
    esac
    
    # Verify installation was successful and fix common pip issues
    echo "🔍 Verifying system dependencies installation..."
    
    # Special handling for pip3 symlink issues
    if ! command_exists pip3 && command_exists pip && ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        echo "ℹ️ pip3 command not found but pip works with ${PYTHON_CMD:-python3}. This is normal on some systems."
    elif ! command_exists pip3 && ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        echo "ℹ️ pip3 command not found but ${PYTHON_CMD:-python3} -m pip works. This is normal on some systems."
    fi
    
    # Final dependency check with more lenient pip detection
    check_system_deps
    local final_status=$?
    
    if [ $final_status -eq 1 ]; then
        # Still failing - try to provide more helpful error information
        echo "❌ System dependency installation verification failed."
        echo ""
        echo "🔍 Debugging information:"
        echo "   - pip3 command: $(command_exists pip3 && echo "✅ Available" || echo "❌ Missing")"
        echo "   - pip command: $(command_exists pip && echo "✅ Available" || echo "❌ Missing")"
        echo "   - ${PYTHON_CMD:-python3} -m pip: $(${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing")"
        echo ""
        echo "If pip is installed but pip3 command is missing, you can create a symlink:"
        echo "   sudo ln -sf \$(which pip) /usr/local/bin/pip3"
        echo ""
        echo "Or the installer will use 'python3 -m pip' instead of 'pip3' command."
        
        # Don't exit - allow installer to continue with python3 -m pip
        echo "⚠️ Continuing installation with available pip interface..."
    elif [ $final_status -eq 2 ]; then
        echo "✅ Essential dependencies verified (build tools will be provided by conda)"
    else
        echo "✅ All system dependencies verified successfully"
    fi
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
    
    # Check if we need conda for build tools (to avoid sudo)
    if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda to provide build tools and avoid sudo requirements"
    # Check command line flags
    elif [ "$FORCE_VENV" = true ]; then
        PYTHON_METHOD="system"
        echo "ℹ️ Using virtual environment (--force-venv)"
    elif [ "$FORCE_CONDA" = true ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda (--force-conda)"
    # Check environment variables
    elif [ "${KWAAINET_USE_SYSTEM_PYTHON:-}" = "true" ]; then
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
        elif ${PYTHON_CMD:-python3} --version >/dev/null 2>&1; then
            PYTHON_VERSION=$(${PYTHON_CMD:-python3} --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            # Use simple version comparison instead of bc
            MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
            MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
            if [ "$MAJOR" -gt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -ge 8 ]; then
                PYTHON_METHOD="system"
                echo "✅ Using system Python $PYTHON_VERSION (${PYTHON_CMD:-python3})"
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

# Function to configure CUDA library paths for bitsandbytes
configure_cuda_paths() {
    echo "🔧 Configuring CUDA library paths for bitsandbytes..."
    
    # Only configure for NVIDIA GPUs
    if [ "$GPU_TYPE" != "nvidia" ]; then
        echo "ℹ️ No NVIDIA GPU detected, skipping CUDA configuration"
        return 0
    fi
    
    # Common CUDA library locations to search
    CUDA_SEARCH_PATHS=(
        "/usr/local/cuda*/lib64"
        "/usr/local/cuda/lib64"
        "/opt/cuda*/lib64"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64"
        "/lib/x86_64-linux-gnu"
        "/lib64"
        "$HOME/.conda/envs/*/lib"
        "$HOME/miniconda3/envs/*/lib"
        "$HOME/anaconda3/envs/*/lib"
    )
    
    CUDA_LIBS_FOUND=()
    echo "🔍 Searching for CUDA libraries..."
    
    # Search for libcudart.so
    for search_path in "${CUDA_SEARCH_PATHS[@]}"; do
        # Handle wildcards in paths
        for expanded_path in $search_path; do
            if [ -d "$expanded_path" ]; then
                if find "$expanded_path" -name "libcudart.so*" -type f 2>/dev/null | head -1 | read -r lib_path; then
                    LIB_DIR=$(dirname "$lib_path")
                    # Add to array if not already present
                    if [[ ! " ${CUDA_LIBS_FOUND[@]} " =~ " ${LIB_DIR} " ]]; then
                        CUDA_LIBS_FOUND+=("$LIB_DIR")
                        echo "✅ Found CUDA libraries in: $LIB_DIR"
                    fi
                fi
            fi
        done
    done
    
    # If no CUDA libraries found, provide guidance
    if [ ${#CUDA_LIBS_FOUND[@]} -eq 0 ]; then
        echo "⚠️ No CUDA libraries found. bitsandbytes may not work with GPU acceleration."
        echo "   To fix this issue:"
        echo "   1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
        echo "   2. Or use conda-forge CUDA packages: 'conda install cuda -c conda-forge'"
        echo "   3. Then run: 'find / -name libcudart.so* 2>/dev/null'"
        echo "   4. Add the directory to LD_LIBRARY_PATH manually"
        return 1
    fi
    
    # Create LD_LIBRARY_PATH export string
    CUDA_PATHS_STR=""
    for lib_path in "${CUDA_LIBS_FOUND[@]}"; do
        if [ -z "$CUDA_PATHS_STR" ]; then
            CUDA_PATHS_STR="$lib_path"
        else
            CUDA_PATHS_STR="$CUDA_PATHS_STR:$lib_path"
        fi
    done
    
    # Update shell configuration files
    echo "📝 Adding CUDA library paths to shell configuration..."
    
    # Get shell configuration files
    SHELL_FILES=(
        "$HOME/.zshrc"
        "$HOME/.bashrc" 
        "$HOME/.bash_profile"
        "$HOME/.profile"
    )
    
    CUDA_EXPORT_LINE="export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$CUDA_PATHS_STR\""
    CUDA_UPDATED=false
    
    for rc_file in "${SHELL_FILES[@]}"; do
        if [ -f "$rc_file" ]; then
            # Check if any LD_LIBRARY_PATH line contains our CUDA paths
            if ! grep -q "LD_LIBRARY_PATH.*$(echo "$CUDA_PATHS_STR" | head -c 20)" "$rc_file"; then
                echo "📝 Adding CUDA paths to $rc_file"
                add_line_if_not_exists "$rc_file" "$CUDA_EXPORT_LINE" "# CUDA library paths for bitsandbytes (added by KwaaiNet installer)"
                CUDA_UPDATED=true
            else
                echo "✅ CUDA paths already configured in $rc_file"
            fi
        fi
    done
    
    # If path wasn't in any existing file, create/update the default for current shell
    if [ "$CUDA_UPDATED" = false ]; then
        SHELL_RC=$(ensure_shell_config)
        echo "📝 Adding CUDA paths to $SHELL_RC (default)"
        add_line_if_not_exists "$SHELL_RC" "$CUDA_EXPORT_LINE" "# CUDA library paths for bitsandbytes (added by KwaaiNet installer)"
    fi
    
    # Update current session LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATHS_STR"
    
    echo "✅ CUDA library paths configured successfully"
    echo "   Added paths: $CUDA_PATHS_STR"
    
    return 0
}

# Function to install tokenizers with fallback handling
install_tokenizers_with_fallback() {
    echo "🔤 Installing tokenizers with build fallback handling..."
    
    # Debug: Show environment status
    echo "🔍 Debug: Checking build environment..."
    echo "   - Rust compiler: $(command -v rustc >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing")"
    echo "   - GCC compiler: $(command -v gcc >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing")"
    echo "   - NO_BUILD_TOOLS flag: $NO_BUILD_TOOLS"
    
    # Ensure Rust environment is available if installed
    if [ -f "$HOME/.cargo/env" ]; then
        echo "🦀 Sourcing Rust environment..."
        source "$HOME/.cargo/env"
    fi
    
    # Add cargo bin to PATH for this session
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "🔍 Added ~/.cargo/bin to PATH"
    fi
    
    # Re-check after sourcing Rust
    if command -v rustc >/dev/null 2>&1; then
        echo "✅ Rust compiler now available: $(rustc --version 2>/dev/null || echo 'version unknown')"
    fi
    
    # Try to install tokenizers with different strategies
    local binary_flag=""
    if [ "$NO_BUILD_TOOLS" = true ]; then
        binary_flag="--only-binary=all"
        echo "ℹ️ Using pre-built wheels only (--no-build-tools)"
    fi
    
    # Strategy 1: Force specific tokenizers version with pre-built wheels first
    echo "📦 Attempting to install tokenizers 0.19.1 (pre-built wheels only)..."
    if $PIP_EXEC install --only-binary=tokenizers "tokenizers==0.19.1" 2>/dev/null; then
        echo "✅ tokenizers 0.19.1 installed successfully (pre-built wheels)"
        return 0
    else
        echo "⚠️ Failed to install tokenizers 0.19.1 pre-built wheels."
    fi
    
    # Strategy 2: Try latest with pre-built wheels only
    echo "📦 Attempting to install latest tokenizers (pre-built wheels only)..."
    if $PIP_EXEC install --only-binary=tokenizers tokenizers 2>/dev/null; then
        echo "✅ tokenizers installed successfully (pre-built wheels)"
        return 0
    else
        echo "⚠️ Failed to install latest tokenizers pre-built wheels."
    fi
    
    # Strategy 3: Try compilation only if Rust is available and build tools allowed
    if [ "$NO_BUILD_TOOLS" != true ] && (command -v rustc >/dev/null 2>&1 || [ -f "$HOME/.cargo/bin/rustc" ]); then
        echo "📦 Attempting to install tokenizers (may compile from source with Rust)..."
        if timeout 300 $PIP_EXEC install tokenizers --no-cache-dir; then
            echo "✅ tokenizers installed successfully (compiled from source)"
            return 0
        else
            echo "⚠️ Failed to compile tokenizers from source."
        fi
    fi
    
    # Strategy 3: Try with specific version that has more wheel support
    echo "📦 Attempting to install older tokenizers version with better wheel support..."
    if $PIP_EXEC install --only-binary=tokenizers "tokenizers==0.19.1" 2>/dev/null; then
        echo "✅ tokenizers 0.19.1 installed successfully (pre-built wheels)"
        return 0
    else
        echo "⚠️ Failed to install tokenizers 0.19.1 pre-built wheels."
    fi
    
    # Strategy 4: Emergency fallback - install via conda if available
    if command -v conda >/dev/null 2>&1 && [ "${PYTHON_METHOD:-}" = "conda" ]; then
        echo "📦 Emergency fallback: trying conda installation..."
        if conda install -y tokenizers -c conda-forge 2>/dev/null; then
            echo "✅ tokenizers installed via conda"
            return 0
        else
            echo "⚠️ Conda installation also failed"
        fi
    fi
    
    # Strategy 5: Last resort - try without any constraints but with timeout
    echo "📦 Last attempt: installing tokenizers with extended timeout..."
    if timeout 600 $PIP_EXEC install tokenizers --no-cache-dir; then
        echo "✅ tokenizers installed successfully (extended timeout)"
        return 0
    else
        echo "❌ All tokenizers installation strategies failed."
        echo ""
        echo "🔧 IMMEDIATE WORKAROUND:"
        echo "   Run installer with: --no-build-tools flag"
        echo "   Command: curl -fsSL ... | bash -s -- --no-build-tools"
        echo ""
        echo "🔧 Manual fix options:"
        echo "   1. Install Rust compiler: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "   2. Install build dependencies: sudo apt-get install build-essential (Ubuntu/Debian)"
        echo "   3. Force pre-built wheels: pip install --only-binary=tokenizers tokenizers"
        echo "   4. Use conda environment: conda install tokenizers -c conda-forge"
        echo ""
        echo "⚠️ Installation will continue, but tokenizers may not work properly."
        return 1
    fi
}

# Main installation flow starts here
echo "🔍 Detecting system configuration..."

# Detect distribution
detect_distro

# Check root privileges
check_root

# Detect GPU
detect_gpu

# Install system dependencies (unless skipped)
if [ "$SKIP_SYSTEM_PACKAGES" = true ]; then
    echo "⏭️ Skipping system package installation"
    # Still check if we have the required dependencies
    if ! check_system_deps; then
        echo "❌ Error: Required system dependencies are missing."
        echo "Install them manually or run without --no-system-packages flag."
        exit 1
    fi
else
    install_system_deps
fi

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
        if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
            echo "📦 Installing build tools via conda to avoid sudo requirements..."
            conda create -y -n kwaainet python=3.10 gcc_linux-64 gxx_linux-64 make
            echo "✅ Created Python 3.10 environment with build tools for KwaaiNet"
        else
            conda create -y -n kwaainet python=3.10
            echo "✅ Created Python 3.10 environment for KwaaiNet"
        fi
    else
        echo "✅ Using existing kwaainet environment"
        # Add build tools if needed and missing
        if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
            echo "📦 Adding build tools to existing conda environment..."
            conda install -y -n kwaainet gcc_linux-64 gxx_linux-64 make || echo "⚠️ Some build tools may already be installed"
        fi
    fi

    # Configure conda channels to avoid Terms of Service issues
    echo "🔧 Configuring conda channels..."
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    # Remove problematic Anaconda commercial channels if they exist
    conda config --env --remove channels https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda config --env --remove channels https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    echo "✅ Configured conda-forge as primary channel (avoids Terms of Service issues)"
    
    # Determine conda installation path
    CONDA_BASE=""
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif command_exists conda; then
        # Try to find conda base from existing installation
        CONDA_BASE=$(conda info --base 2>/dev/null || dirname $(dirname $(which conda)) 2>/dev/null)
    fi
    
    if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "❌ Error: Cannot find conda installation or conda.sh script"
        exit 1
    fi
    
    echo "ℹ️ Using conda installation at: $CONDA_BASE"
    
    # Source conda for current session
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    
    # Activate environment
    if ! conda activate kwaainet 2>/dev/null; then
        echo "⚠️ Failed to activate kwaainet environment. Re-initializing conda..."
        "$CONDA_BASE/bin/conda" init bash
        "$CONDA_BASE/bin/conda" init zsh 2>/dev/null || true
        # Re-source conda
        . "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate kwaainet
    fi
    PYTHON_EXEC="python"
    PIP_EXEC="$(get_pip_command)"
    
elif [ "$PYTHON_METHOD" = "system" ]; then
    # Use system Python with virtual environment
    echo "⚙️ Setting up KwaaiNet virtual environment..."
    
    VENV_PATH="$HOME/.kwaainet-venv"
    if [ ! -d "$VENV_PATH" ]; then
        ${PYTHON_CMD:-python3} -m venv "$VENV_PATH"
        echo "✅ Created virtual environment for KwaaiNet"
    else
        echo "✅ Using existing virtual environment"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    PYTHON_EXEC="$VENV_PATH/bin/python"
    PIP_EXEC="$(get_pip_command)"
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

# Install Petals
if [ "$NO_BUILD_TOOLS" = true ]; then
    echo "📦 Installing Petals from PyPI (pre-built wheels only)..."
    if $PIP_EXEC install petals --only-binary=all 2>/dev/null; then
        echo "✅ Petals installed from PyPI (pre-built)"
    else
        echo "⚠️ No pre-built Petals available. Trying regular PyPI installation..."
        if $PIP_EXEC install petals 2>/dev/null; then
            echo "✅ Petals installed from PyPI"
        else
            echo "❌ Failed to install Petals without build tools"
            echo "Try running without --no-build-tools or install build dependencies manually"
            exit 1
        fi
    fi
else
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
fi

# Install tokenizers first (handles Rust compilation issues)
install_tokenizers_with_fallback

# Install compatible versions of transformers and huggingface_hub
echo "📦 Installing compatible transformers and huggingface_hub versions..."

# Check Python version for compatibility
PYTHON_VERSION=$(${PYTHON_CMD:-python3} -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

# Add --only-binary flag if no build tools
BINARY_FLAG=""
if [ "$NO_BUILD_TOOLS" = true ]; then
    BINARY_FLAG="--only-binary=all"
    echo "ℹ️ Using pre-built wheels only (--no-build-tools)"
fi

# Ensure Rust environment is available for any compilation
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 7 ]; then
    echo "📦 Using Python 3.7 compatible versions..."
    # Try transformers 4.21.3 which was the last version with good Python 3.7 support
    if $PIP_EXEC install $BINARY_FLAG "transformers==4.21.3" "huggingface_hub>=0.8.0,<0.20.0"; then
        echo "✅ Successfully installed Python 3.7 compatible transformers and huggingface_hub"
    else
        echo "⚠️ Failed to install Python 3.7 compatible versions. Trying default versions..."
        if $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1" "huggingface_hub>=0.20.0"; then
            echo "✅ Successfully installed default transformers and huggingface_hub"
        else
            echo "⚠️ Failed to install transformers/huggingface_hub. May have compatibility issues..."
        fi
    fi
else
    # Python 3.8+ - use the secure version with tokenizers pinning
    echo "📦 Installing transformers with tokenizers dependency pinning..."
    if $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1" "tokenizers>=0.19.0,<0.20.0" "huggingface_hub>=0.20.0"; then
        echo "✅ Successfully installed compatible transformers and huggingface_hub"
    else
        echo "⚠️ Failed to install transformers/huggingface_hub. Trying without tokenizers pinning..."
        if $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1" "huggingface_hub>=0.20.0"; then
            echo "✅ Successfully installed compatible transformers and huggingface_hub (fallback)"
        else
            echo "⚠️ Failed to install transformers/huggingface_hub. May have compatibility issues..."
        fi
    fi
fi

# Install from the local development version
INSTALLER_DIR="$(dirname "$0")"
if [ -d "$INSTALLER_DIR/linux" ]; then
    echo "📦 Installing from local development version..."
    if $PIP_EXEC install -e "$INSTALLER_DIR/linux/" 2>/dev/null; then
        echo "✅ KwaaiNet Linux package installed successfully (local development version)"
    else
        echo "⚠️ Failed to install local development version. Installing from GitHub..."
        
        # Install PyTorch based on GPU availability
        if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
            echo "📦 Installing PyTorch (CUDA version for NVIDIA GPU)..."
            echo "   This may take a few minutes to download..."
            if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
                echo "✅ PyTorch CUDA installed successfully"
            else
                echo "⚠️ Failed to install CUDA PyTorch. Falling back to CPU version..."
                if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                    echo "✅ PyTorch CPU installed successfully"
                else
                    echo "❌ Failed to install PyTorch. Please check your internet connection."
                    exit 1
                fi
            fi
        else
            echo "📦 Installing PyTorch (CPU version)..."
            echo "   This may take a few minutes to download..."
            if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                echo "✅ PyTorch CPU installed successfully"
            else
                echo "❌ Failed to install PyTorch. Please check your internet connection."
                exit 1
            fi
        fi
        
        # Install bitsandbytes for quantization support
        echo "📦 Installing bitsandbytes for quantization support..."
        if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
            echo "   Installing CUDA-compatible version for NVIDIA GPU..."
            # First try standard installation which should auto-detect CUDA
            if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
                echo "✅ bitsandbytes CUDA installed successfully"
            else
                echo "⚠️ Failed to install bitsandbytes CUDA. Quantization may not work properly."
            fi
        else
            echo "   Installing CPU version..."
            if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
                echo "✅ bitsandbytes CPU installed successfully"
            else
                echo "⚠️ Failed to install bitsandbytes. Quantization may not work properly."
            fi
        fi
        
        # Install KwaaiNet Linux package from GitHub as fallback
        echo "📦 Installing KwaaiNet Linux package from GitHub..."
        if $PIP_EXEC install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/linux" 2>/dev/null; then
            echo "✅ KwaaiNet Linux package installed successfully"
        else
            echo "❌ Failed to install KwaaiNet Linux package from GitHub"
            echo "Please check your internet connection and try again."
            exit 1
        fi
    fi
else
    echo "⚠️ Local development version not found. Installing from GitHub repository..."
    
    # Install PyTorch based on GPU availability
    if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
        echo "📦 Installing PyTorch (CUDA version for NVIDIA GPU)..."
        echo "   This may take a few minutes to download..."
        if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
            echo "✅ PyTorch CUDA installed successfully"
        else
            echo "⚠️ Failed to install CUDA PyTorch. Falling back to CPU version..."
            if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                echo "✅ PyTorch CPU installed successfully"
            else
                echo "❌ Failed to install PyTorch. Please check your internet connection."
                exit 1
            fi
        fi
    else
        echo "📦 Installing PyTorch (CPU version)..."
        echo "   This may take a few minutes to download..."
        if $PIP_EXEC install $BINARY_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            echo "✅ PyTorch CPU installed successfully"
        else
            echo "❌ Failed to install PyTorch. Please check your internet connection."
            exit 1
        fi
    fi
    
    # Install bitsandbytes for quantization support
    echo "📦 Installing bitsandbytes for quantization support..."
    if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
        echo "   Installing CUDA-compatible version for NVIDIA GPU..."
        # First try standard installation which should auto-detect CUDA
        if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
            echo "✅ bitsandbytes CUDA installed successfully"
        else
            echo "⚠️ Failed to install bitsandbytes CUDA. Quantization may not work properly."
        fi
    else
        echo "   Installing CPU version..."
        if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
            echo "✅ bitsandbytes CPU installed successfully"
        else
            echo "⚠️ Failed to install bitsandbytes. Quantization may not work properly."
        fi
    fi
    
    # Install KwaaiNet Linux package from GitHub
    echo "📦 Installing KwaaiNet Linux package..."
    if $PIP_EXEC install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/linux" 2>/dev/null; then
        echo "✅ KwaaiNet Linux package installed successfully"
    else
        echo "❌ Failed to install KwaaiNet Linux package from GitHub"
        echo "Please check your internet connection and try again."
        exit 1
    fi
fi

# Configure CUDA library paths for bitsandbytes (NVIDIA GPUs only)
configure_cuda_paths

# Install KwaaiNet Linux package
echo "📦 Installing KwaaiNet for Linux..."

# Clear cached versions and uninstall existing packages to avoid conflicts
echo "🧹 Clearing any cached versions and removing existing packages..."
$PIP_EXEC cache remove kwaainet-linux &>/dev/null || true
$PIP_EXEC cache remove kwaainet_linux &>/dev/null || true
$PIP_EXEC uninstall -y kwaainet-linux kwaainet_linux &>/dev/null || true

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
echo "📦 Installing KwaaiNet for Linux in development mode..."
cd "$PROJECT_PATH/Installer/linux"
$PIP_EXEC install -e .

if [ $? -eq 0 ]; then
    echo "✅ KwaaiNet Linux package installed successfully"
else
    echo "❌ Failed to install KwaaiNet Linux package"
    exit 1
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

# Find conda installation (prioritize user installations over system)
CONDA_PATH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
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
    exit 1
fi

# Source conda without changing the prompt
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Configure CUDA library paths for bitsandbytes if not already set
if command -v nvidia-smi >/dev/null 2>&1; then
    # Search for CUDA libraries in common locations
    CUDA_PATHS=""
    for search_path in "/usr/local/cuda*/lib64" "/usr/local/cuda/lib64" "/opt/cuda*/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "$CONDA_PATH/envs/kwaainet/lib"; do
        for expanded_path in $search_path; do
            if [ -d "$expanded_path" ] && find "$expanded_path" -name "libcudart.so*" -type f >/dev/null 2>&1; then
                if [ -z "$CUDA_PATHS" ]; then
                    CUDA_PATHS="$expanded_path"
                else
                    CUDA_PATHS="$CUDA_PATHS:$expanded_path"
                fi
            fi
        done
    done
    
    if [ -n "$CUDA_PATHS" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATHS"
    fi
fi

# Activate the environment and run the command
conda activate kwaainet

# Try to run the module, with fallback to direct execution
if ! python -m kwaainet.runner "$@" 2>/dev/null; then
    # Fallback: try running from the source installation directory
    if [ -d "$HOME/.kwaainet/source" ]; then
        echo "⚠️ Module import failed, trying fallback from source directory..."
        SOURCE_DIR=\$(find "$HOME/.kwaainet/source" -name "OpenAI-Petal*" -type d | head -1)
        if [ -n "\$SOURCE_DIR" ] && [ -f "\$SOURCE_DIR/Installer/linux/kwaainet/runner.py" ]; then
            export PYTHONPATH="\$SOURCE_DIR/Installer/linux:\$PYTHONPATH"
            python -m kwaainet.runner "$@"
        else
            echo "❌ Error: Could not find KwaaiNet installation. Please run the installer again."
            exit 1
        fi
    else
        echo "❌ Error: KwaaiNet module not found. Please run the installer again."
        exit 1
    fi
fi
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

# Configure CUDA library paths for bitsandbytes if not already set
if command -v nvidia-smi >/dev/null 2>&1; then
    # Search for CUDA libraries in common locations
    CUDA_PATHS=""
    for search_path in "/usr/local/cuda*/lib64" "/usr/local/cuda/lib64" "/opt/cuda*/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib64"; do
        for expanded_path in \$search_path; do
            if [ -d "\$expanded_path" ] && find "\$expanded_path" -name "libcudart.so*" -type f >/dev/null 2>&1; then
                if [ -z "\$CUDA_PATHS" ]; then
                    CUDA_PATHS="\$expanded_path"
                else
                    CUDA_PATHS="\$CUDA_PATHS:\$expanded_path"
                fi
            fi
        done
    done
    
    if [ -n "\$CUDA_PATHS" ]; then
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CUDA_PATHS"
    fi
fi

# Activate virtual environment and run the command
source "\$VENV_PATH/bin/activate"

# Try to run the module, with fallback to direct execution
if ! python -m kwaainet.runner "\$@" 2>/dev/null; then
    # Fallback: try running from the source installation directory
    if [ -d "\$HOME/.kwaainet/source" ]; then
        echo "⚠️ Module import failed, trying fallback from source directory..."
        SOURCE_DIR=\$(find "\$HOME/.kwaainet/source" -name "OpenAI-Petal*" -type d | head -1)
        if [ -n "\$SOURCE_DIR" ] && [ -f "\$SOURCE_DIR/Installer/linux/kwaainet/runner.py" ]; then
            export PYTHONPATH="\$SOURCE_DIR/Installer/linux:\$PYTHONPATH"
            python -m kwaainet.runner "\$@"
        else
            echo "❌ Error: Could not find KwaaiNet installation. Please run the installer again."
            exit 1
        fi
    else
        echo "❌ Error: KwaaiNet module not found. Please run the installer again."
        exit 1
    fi
fi
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
    # Ensure conda is sourced and environment is active
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
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
if [ "$GPU_TYPE" = "nvidia" ]; then
echo ""
echo "🔧 NVIDIA GPU detected - CUDA library paths have been configured."
echo "   If you encounter bitsandbytes CUDA errors:"
echo "   1. Restart your shell or run: source ~/.bashrc"
echo "   2. Verify CUDA installation: nvidia-smi"
echo "   3. Check library paths: echo \$LD_LIBRARY_PATH"
fi
echo ""
echo "📚 For more information, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"
echo "=========================================================="