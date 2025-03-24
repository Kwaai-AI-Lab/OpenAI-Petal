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
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    
    # Remove conda environment if it exists
    if conda info --envs | grep -q kwaainet; then
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

echo ""
echo "=========================================================="
echo "✅ KwaaiNet has been successfully uninstalled from your system!"
echo "=========================================================="