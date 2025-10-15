#!/bin/bash
# Detect current development environment and output the appropriate environment file

HOSTNAME=$(hostname)
OS=$(uname -s)

# Detect based on hostname and OS
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    if [[ "$HOSTNAME" =~ rezarassool ]] || [[ "$HOSTNAME" =~ "Rezas-Mac" ]]; then
        echo ".claude/environments/macos-rezarassool.md"
    else
        echo ".claude/environments/macos-unknown.md (CREATE THIS FILE)"
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Linux
    if [[ "$HOSTNAME" =~ metro ]]; then
        echo ".claude/environments/linux-metro.md"
    else
        echo ".claude/environments/linux-unknown.md (CREATE THIS FILE)"
    fi
elif [[ "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
    # Windows
    echo ".claude/environments/windows-unknown.md (CREATE THIS FILE)"
else
    echo "Unknown environment: $OS on $HOSTNAME"
fi
