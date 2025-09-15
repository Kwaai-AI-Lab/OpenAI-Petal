#!/bin/bash

# Version Update Script for OpenAI-Petal
# Usage: ./scripts/update_version.sh [new_version]
# Example: ./scripts/update_version.sh 0.2.17

set -e

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <new_version>"
    echo "📖 Example: $0 0.2.17"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔄 Updating version to $NEW_VERSION..."

# Update VERSION file
echo "$NEW_VERSION" > "$PROJECT_ROOT/VERSION"
echo "✅ Updated VERSION file"

# Update Linux installer
sed -i.bak "s/# KwaaiNet for Linux - One-Step Installer v[0-9.]*/# KwaaiNet for Linux - One-Step Installer v$NEW_VERSION/" "$PROJECT_ROOT/Installer/linux/linuxinstaller.sh"
sed -i.bak "s/INSTALLER_VERSION=\"[0-9.]*\"/INSTALLER_VERSION=\"$NEW_VERSION\"/" "$PROJECT_ROOT/Installer/linux/linuxinstaller.sh"
rm -f "$PROJECT_ROOT/Installer/linux/linuxinstaller.sh.bak"
echo "✅ Updated Linux installer"

# Update macOS installer if it exists
if [ -f "$PROJECT_ROOT/Installer/macOS/macinstaller.sh" ]; then
    sed -i.bak "s/# KwaaiNet for macOS - One-Step Installer v[0-9.]*/# KwaaiNet for macOS - One-Step Installer v$NEW_VERSION/" "$PROJECT_ROOT/Installer/macOS/macinstaller.sh"
    sed -i.bak "s/INSTALLER_VERSION=\"[0-9.]*\"/INSTALLER_VERSION=\"$NEW_VERSION\"/" "$PROJECT_ROOT/Installer/macOS/macinstaller.sh"
    rm -f "$PROJECT_ROOT/Installer/macOS/macinstaller.sh.bak"
    echo "✅ Updated macOS installer"
fi

# Update README badge
sed -i.bak "s/installer-v[0-9.]*/installer-v$NEW_VERSION/" "$PROJECT_ROOT/README.md"
rm -f "$PROJECT_ROOT/README.md.bak"
echo "✅ Updated README badge"

# Update any package.json if it exists
if [ -f "$PROJECT_ROOT/package.json" ]; then
    sed -i.bak "s/\"version\": \"[0-9.]*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/package.json"
    rm -f "$PROJECT_ROOT/package.json.bak"
    echo "✅ Updated package.json"
fi

echo ""
echo "🎉 Version updated to $NEW_VERSION in all files!"
echo "📝 Next steps:"
echo "   1. git add ."
echo "   2. git commit -m \"Increment version to v$NEW_VERSION\""
echo "   3. git push origin main"
echo "   4. git tag v$NEW_VERSION && git push origin v$NEW_VERSION"