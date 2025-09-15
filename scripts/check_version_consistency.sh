#!/bin/bash

# Version Consistency Checker
# This script verifies all version numbers are in sync

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Checking version consistency..."

# Read the canonical version from VERSION file
if [ ! -f "$PROJECT_ROOT/VERSION" ]; then
    echo "❌ VERSION file not found"
    exit 1
fi

CANONICAL_VERSION=$(cat "$PROJECT_ROOT/VERSION" | tr -d '\n\r')
echo "📋 Canonical version: $CANONICAL_VERSION"

# Check all files that should have matching versions
files_to_check=(
    "README.md:installer-v([0-9.]+)"
    "Installer/linux/linuxinstaller.sh:INSTALLER_VERSION=\"([0-9.]+)\""
)

errors=0

for entry in "${files_to_check[@]}"; do
    file="${entry%%:*}"
    pattern="${entry##*:}"
    if [ -f "$PROJECT_ROOT/$file" ]; then
        found_version=$(grep -oE "$pattern" "$PROJECT_ROOT/$file" | head -1 | grep -oE "[0-9.]+")

        if [ "$found_version" = "$CANONICAL_VERSION" ]; then
            echo "✅ $file: $found_version"
        else
            echo "❌ $file: $found_version (expected $CANONICAL_VERSION)"
            errors=$((errors + 1))
        fi
    else
        echo "⚠️ $file: not found"
    fi
done

if [ $errors -gt 0 ]; then
    echo ""
    echo "❌ Version inconsistency detected! Run: ./scripts/update_version.sh $CANONICAL_VERSION"
    exit 1
else
    echo ""
    echo "✅ All versions are consistent!"
fi