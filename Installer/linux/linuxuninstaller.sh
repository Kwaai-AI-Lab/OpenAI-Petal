#!/bin/bash

# KwaaiNet for Linux - Uninstaller
# This script completely removes KwaaiNet and its environment from your system

set -e  # Exit on error

echo "=========================================================="
echo "KwaaiNet for Linux - Uninstaller"
echo "=========================================================="
echo "This will remove KwaaiNet and its environment from your system."
echo ""
read -p "Are you sure you want to uninstall KwaaiNet? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        USE_SUDO=""
    else
        USE_SUDO="sudo"
    fi
}

check_root

# Remove launcher scripts
echo "🧹 Removing launcher scripts..."
if [ -f "$HOME/.local/bin/kwaainet" ]; then
    rm "$HOME/.local/bin/kwaainet"
    echo "  ✓ Removed ~/.local/bin/kwaainet"
fi

if [ -f "/usr/local/bin/kwaainet" ]; then
    if [ -w "/usr/local/bin" ]; then
        rm "/usr/local/bin/kwaainet"
        echo "  ✓ Removed /usr/local/bin/kwaainet"
    else
        echo "  ⚠️ Need sudo permission to remove /usr/local/bin/kwaainet"
        $USE_SUDO rm "/usr/local/bin/kwaainet" 2>/dev/null || true
        echo "  ✓ Removed /usr/local/bin/kwaainet"
    fi
fi

# Check if conda exists and remove environment
if command_exists conda; then
    echo "🗑️ Checking for kwaainet conda environment..."
    
    # Source conda with better error handling
    CONDA_BASE=""
    if conda info --base >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
    
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "  ⚠️ Could not source conda environment. Trying direct removal..."
    fi
    
    # Remove conda environment if it exists
    if conda info --envs 2>/dev/null | grep -q kwaainet; then
        echo "🗑️ Removing kwaainet conda environment..."
        conda deactivate 2>/dev/null || true
        if conda env remove -n kwaainet -y 2>/dev/null; then
            echo "  ✓ Environment removed"
        else
            echo "  ⚠️ Failed to remove environment automatically. You may need to run: conda env remove -n kwaainet -y"
        fi
    else
        echo "  ✓ No kwaainet environment found"
    fi
else
    echo "  ✓ Conda not found, skipping environment removal"
fi

# Remove virtual environment if it exists
VENV_PATH="$HOME/.kwaainet-venv"
if [ -d "$VENV_PATH" ]; then
    echo "🗑️ Removing kwaainet virtual environment..."
    rm -rf "$VENV_PATH"
    echo "  ✓ Removed $VENV_PATH"
else
    echo "  ✓ No virtual environment found"
fi

# Remove installed packages using pip
echo "🧹 Removing kwaainet packages..."
if command_exists pip; then
    if pip uninstall kwaainet-linux kwaainet_linux -y 2>/dev/null; then
        echo "  ✓ Removed kwaainet packages with pip"
    else
        echo "  ✓ No kwaainet packages found with pip"
    fi
fi

if command_exists pip3; then
    if pip3 uninstall kwaainet-linux kwaainet_linux -y 2>/dev/null; then
        echo "  ✓ Removed kwaainet packages with pip3"
    else
        echo "  ✓ No kwaainet packages found with pip3"
    fi
fi

# Remove cache directories
echo "🧹 Cleaning up cache directories..."
CACHE_DIRS=(
    "$HOME/.cache/huggingface"
    "$HOME/.cache/tf-cache"
    "$HOME/.cache/llama-index-cache"
    "$HOME/.cache/nltk-cache"
    "$HOME/.cache/tiktoken-cache"
    "$HOME/.cache/temp"
    "$HOME/.cache/pip"
    "$HOME/.cache/torch"
    "$HOME/.cache/transformers"
)

for dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        read -p "Remove cache directory $dir? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo "  ✓ Removed $dir"
        else
            echo "  ✓ Skipped $dir"
        fi
    fi
done

# Remove configuration directory
if [ -d "$HOME/.kwaainet" ]; then
    echo "🧹 Removing KwaaiNet configuration..."
    rm -rf "$HOME/.kwaainet"
    echo "  ✓ Removed ~/.kwaainet"
fi

# Clean pip cache
if command_exists pip; then
    echo "🧹 Cleaning pip cache..."
    pip cache remove kwaainet-linux 2>/dev/null || echo "  ✓ No kwaainet-linux cache found"
    pip cache remove kwaainet_linux 2>/dev/null || echo "  ✓ No kwaainet_linux cache found"
    pip cache remove petals 2>/dev/null || echo "  ✓ No petals cache found"
    echo "  ✓ Pip cache cleanup completed"
fi

# Clean up shell configuration files
echo "🧹 Cleaning up shell configuration entries..."
SHELL_FILES=(
    "$HOME/.zshrc"
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.profile"
)

for rc_file in "${SHELL_FILES[@]}"; do
    if [ -f "$rc_file" ]; then
        echo "  Checking $rc_file..."
        
        # Create a backup before modifying
        cp "$rc_file" "$rc_file.kwaaibak"
        
        # Remove PATH entry with KwaaiNet comment
        if grep -q "# Added by KwaaiNet installer" "$rc_file"; then
            # Use sed to remove the comment and the following line
            sed -i.bak '/# Added by KwaaiNet installer/{N;d;}' "$rc_file"
            echo "  ✓ Removed KwaaiNet PATH from $rc_file"
        elif grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$rc_file"; then
            # Ask user if they want to remove the PATH export (might be used by other tools)
            echo "  Found PATH export in $rc_file that may be used by other tools."
            read -p "  Remove export PATH=\"\$HOME/.local/bin:\$PATH\"? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sed -i.bak '/export PATH="$HOME\/.local\/bin:$PATH"/d' "$rc_file"
                echo "  ✓ Removed PATH export from $rc_file"
            else
                echo "  ✓ Kept PATH export in $rc_file"
            fi
        fi
        
        # If backup was created successfully, remove it
        if [ -f "$rc_file.bak" ]; then
            rm "$rc_file.bak"
        fi
    fi
done

# Ask if the user wants to remove conda initialization
if command_exists conda; then
    echo ""
    echo "KwaaiNet may have added conda initialization to your shell configuration files."
    read -p "Would you like to remove conda initialization as well? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CONDA_REMOVED=false
        for rc_file in "${SHELL_FILES[@]}"; do
            if [ -f "$rc_file" ] && grep -q "# >>> conda initialize >>>" "$rc_file"; then
                # Create a backup before modifying
                if cp "$rc_file" "$rc_file.conda.bak" 2>/dev/null; then
                    # Remove conda initialization block
                    if sed -i.bak '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$rc_file" 2>/dev/null; then
                        echo "  ✓ Removed conda initialization from $rc_file"
                        CONDA_REMOVED=true
                    else
                        echo "  ⚠️ Failed to remove conda initialization from $rc_file"
                        # Restore from backup if sed failed
                        cp "$rc_file.conda.bak" "$rc_file" 2>/dev/null || true
                    fi
                    
                    # If backup was created successfully, remove it
                    if [ -f "$rc_file.bak" ]; then
                        rm "$rc_file.bak" 2>/dev/null || true
                    fi
                else
                    echo "  ⚠️ Could not create backup for $rc_file, skipping conda removal"
                fi
            fi
        done
        
        if [ "$CONDA_REMOVED" = false ]; then
            echo "  ✓ No conda initialization found in shell files"
        fi
    fi
fi

# Ask if the user wants to uninstall Miniconda (only if it was installed by the installer)
if [ -d "$HOME/miniconda3" ] && command_exists conda; then
    echo ""
    echo "Miniconda installation detected in $HOME/miniconda3."
    echo "This may have been installed by KwaaiNet setup."
    read -p "Would you like to uninstall Miniconda as well? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️ Uninstalling Miniconda..."
        rm -rf "$HOME/miniconda3"
        echo "  ✓ Miniconda removed"
        
        # Remove conda from PATH in all shell files
        for rc_file in "${SHELL_FILES[@]}"; do
            if [ -f "$rc_file" ] && grep -q "miniconda3" "$rc_file"; then
                cp "$rc_file" "$rc_file.miniconda.bak"
                sed -i.bak '/miniconda3/d' "$rc_file"
                echo "  ✓ Removed miniconda3 references from $rc_file"
                if [ -f "$rc_file.bak" ]; then
                    rm "$rc_file.bak"
                fi
            fi
        done
    fi
fi

# Remove any remaining kwaainet processes
echo "🔍 Checking for running kwaainet processes..."
KWAAINET_PIDS=$(pgrep -f kwaainet 2>/dev/null || true)
if [ -n "$KWAAINET_PIDS" ]; then
    echo "Found running kwaainet processes: $KWAAINET_PIDS"
    read -p "Kill running kwaainet processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$KWAAINET_PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        # Force kill if still running
        KWAAINET_PIDS=$(pgrep -f kwaainet 2>/dev/null || true)
        if [ -n "$KWAAINET_PIDS" ]; then
            echo "$KWAAINET_PIDS" | xargs kill -KILL 2>/dev/null || true
        fi
        echo "  ✓ Stopped kwaainet processes"
    fi
else
    echo "  ✓ No running kwaainet processes found"
fi

# Clean up any systemd services (if they exist)
if command_exists systemctl; then
    KWAAINET_SERVICES=$(systemctl --user list-units --all | grep kwaainet | awk '{print $1}' 2>/dev/null || true)
    if [ -n "$KWAAINET_SERVICES" ]; then
        echo "🗑️ Found kwaainet systemd services..."
        for service in $KWAAINET_SERVICES; do
            systemctl --user stop "$service" 2>/dev/null || true
            systemctl --user disable "$service" 2>/dev/null || true
            echo "  ✓ Stopped and disabled $service"
        done
    fi
fi

echo ""
echo "=========================================================="
echo "✅ KwaaiNet has been successfully uninstalled from your system!"
echo ""
echo "📋 Summary of what was removed:"
echo "  • Launcher scripts (kwaainet command)"
echo "  • Conda environment (if found)"
echo "  • Virtual environment (if found)"
echo "  • Configuration files (~/.kwaainet)"
echo "  • Shell PATH modifications"
echo "  • Cache directories (as selected)"
echo ""
echo "Note: We've created backup files with extension .kwaaibak for any"
echo "shell configuration files that were modified. You can delete these"
echo "if everything is working correctly."
echo ""
echo "⚠️ You may need to restart your terminal or run 'source ~/.bashrc'"
echo "to reload your shell configuration."
echo "=========================================================="