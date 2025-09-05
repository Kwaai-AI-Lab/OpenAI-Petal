#!/bin/bash

# KwaaiNet for Mac - Uninstaller
# This script completely removes KwaaiNet and its environment from your system

set -e  # Exit on error

echo "=========================================================="
echo "KwaaiNet for Mac - Uninstaller"
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
        sudo rm "/usr/local/bin/kwaainet"
        echo "  ✓ Removed /usr/local/bin/kwaainet"
    fi
fi

# Check if conda exists
if command_exists conda; then
    # Source conda
    if [[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    
    # Remove conda environment if it exists
    if conda info --envs 2>/dev/null | grep -q kwaainet; then
        echo "🗑️ Removing kwaainet conda environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n kwaainet -y
        echo "  ✓ Environment removed"
    else
        echo "  ✓ No kwaainet environment found"
    fi
else
    echo "  ✓ Conda not found, skipping environment removal"
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
)

for dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "  ✓ Removed $dir"
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
    pip cache remove kwaainet-mac 2>/dev/null || true
    pip cache remove kwaainet_mac 2>/dev/null || true
    echo "  ✓ Cleaned pip cache"
fi

# Clean up shell configuration files
echo "🧹 Cleaning up shell configuration entries..."
SHELL_FILES=(
    "$HOME/.zshrc"
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
)

for rc_file in "${SHELL_FILES[@]}"; do
    if [ -f "$rc_file" ]; then
        echo "  Checking $rc_file..."
        
        # Create a backup before modifying
        cp "$rc_file" "$rc_file.kwaaibak"
        
        # Remove PATH entry
        if grep -q "# Added by KwaaiNet installer" "$rc_file"; then
            # Use sed to remove the comment and the following line
            sed -i.bak '/# Added by KwaaiNet installer/{N;d;}' "$rc_file"
            echo "  ✓ Removed KwaaiNet PATH from $rc_file"
        elif grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$rc_file"; then
            # In case the comment is missing but the export exists
            sed -i.bak '/export PATH="$HOME\/.local\/bin:$PATH"/d' "$rc_file"
            echo "  ✓ Removed PATH export from $rc_file"
        fi
        
        # If backup was created successfully, remove it
        if [ -f "$rc_file.bak" ]; then
            rm "$rc_file.bak"
        fi
    fi
done

# Ask if the user wants to remove conda initialization
echo ""
echo "KwaaiNet may have added conda initialization to your shell configuration files."
read -p "Would you like to remove conda initialization as well? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for rc_file in "${SHELL_FILES[@]}"; do
        if [ -f "$rc_file" ] && grep -q "# >>> conda initialize >>>" "$rc_file"; then
            # Create a backup before modifying
            cp "$rc_file" "$rc_file.conda.bak"
            
            # Remove conda initialization block
            sed -i.bak '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$rc_file"
            echo "  ✓ Removed conda initialization from $rc_file"
            
            # If backup was created successfully, remove it
            if [ -f "$rc_file.bak" ]; then
                rm "$rc_file.bak"
            fi
        fi
    done
fi

# Ask if the user wants to uninstall Miniconda
if command_exists brew && brew list --cask | grep -q miniconda; then
    echo ""
    echo "Miniconda was installed by Homebrew as part of KwaaiNet setup."
    read -p "Would you like to uninstall Miniconda as well? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️ Uninstalling Miniconda..."
        brew uninstall --cask miniconda
        echo "  ✓ Miniconda uninstalled"
    fi
fi

# Ask if the user wants to uninstall Homebrew
if command_exists brew; then
    echo ""
    echo "Homebrew may have been installed by KwaaiNet setup."
    read -p "Would you like to uninstall Homebrew completely? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️ Uninstalling Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/uninstall.sh)"
        echo "  ✓ Homebrew uninstall script completed"
    fi
fi

echo ""
echo "=========================================================="
echo "✅ KwaaiNet has been successfully uninstalled from your system!"
echo ""
echo "Note: We've created backup files with extension .kwaaibak for any"
echo "shell configuration files that were modified. You can delete these"
echo "if everything is working correctly."
echo "=========================================================="