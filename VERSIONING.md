# Version Management Guide

This document describes the automated version management system for OpenAI-Petal.

## Quick Start

### Update Version (Recommended)
```bash
# Update to new version automatically
./scripts/update_version.sh 0.2.17

# Check consistency
./scripts/check_version_consistency.sh

# Commit changes
git add .
git commit -m "Increment version to v0.2.17"
git push origin main
git tag v0.2.17 && git push origin v0.2.17
```

### Manual Process (If Scripts Not Available)
```bash
# 1. Update VERSION file
echo "0.2.17" > VERSION

# 2. Update Linux installer
sed -i 's/v0.2.16/v0.2.17/g' Installer/linux/linuxinstaller.sh
sed -i 's/INSTALLER_VERSION="0.2.16"/INSTALLER_VERSION="0.2.17"/' Installer/linux/linuxinstaller.sh

# 3. Update README badge
sed -i 's/installer-v0.2.16/installer-v0.2.17/' README.md

# 4. Commit
git add . && git commit -m "Increment version to v0.2.17"
```

## Version Management System

### Components

1. **VERSION file** - Central source of truth for current version
2. **update_version.sh** - Automated script to update all version references
3. **check_version_consistency.sh** - Validates all versions are in sync
4. **Pre-commit hook** - Prevents commits with inconsistent versions

### Files That Contain Versions

| File | Pattern | Purpose |
|------|---------|---------|
| `VERSION` | `0.2.16` | Canonical version source |
| `README.md` | `installer-v0.2.16` | Badge display |
| `Installer/linux/linuxinstaller.sh` | `INSTALLER_VERSION="0.2.16"` | Installer version |
| `Installer/macOS/macinstaller.sh` | `INSTALLER_VERSION="0.2.16"` | Installer version |

### Version Numbering Scheme

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 0.2.16)
- **MAJOR**: Breaking changes
- **MINOR**: New features, significant improvements
- **PATCH**: Bug fixes, small improvements

### When to Increment Versions

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Critical bug fixes | PATCH | 0.2.16 → 0.2.17 |
| New features | MINOR | 0.2.16 → 0.3.0 |
| Breaking changes | MAJOR | 0.2.16 → 1.0.0 |
| Installer improvements | PATCH | 0.2.16 → 0.2.17 |
| Security fixes | PATCH | 0.2.16 → 0.2.17 |

## Automation Features

### Pre-commit Hook
- **Location**: `.githooks/pre-commit` → `.git/hooks/pre-commit`
- **Purpose**: Prevents commits with version inconsistencies
- **Install**: `cp .githooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`

### Consistency Checking
```bash
# Check if all versions match
./scripts/check_version_consistency.sh

# Output example:
# ✅ README.md: 0.2.16
# ✅ Installer/linux/linuxinstaller.sh: 0.2.16
# ✅ All versions are consistent!
```

### Automated Updates
```bash
# Update all version references at once
./scripts/update_version.sh 0.2.17

# This updates:
# - VERSION file
# - Linux installer
# - macOS installer
# - README badge
# - package.json (if exists)
```

## Git Workflow Integration

### Recommended Workflow
```bash
# 1. Make changes
git add .

# 2. Update version (if needed)
./scripts/update_version.sh 0.2.17

# 3. Commit (pre-commit hook runs automatically)
git commit -m "Add new feature - increment to v0.2.17"

# 4. Push and tag
git push origin main
git tag v0.2.17 && git push origin v0.2.17
```

### Why This System?

**Problems Solved:**
- ❌ Forgetting to update version numbers
- ❌ Inconsistent versions across files
- ❌ Manual process prone to errors
- ❌ No validation of version consistency

**Benefits:**
- ✅ **Automated**: Single command updates everywhere
- ✅ **Consistent**: All files guaranteed to match
- ✅ **Validated**: Pre-commit hooks prevent inconsistencies
- ✅ **Documented**: Clear process for contributors

## Troubleshooting

### Version Inconsistency Error
```bash
❌ Version inconsistency detected!

# Fix with:
./scripts/update_version.sh 0.2.16  # Use canonical version
```

### Pre-commit Hook Failing
```bash
# Check what's wrong:
./scripts/check_version_consistency.sh

# Fix and recommit:
./scripts/update_version.sh 0.2.16
git add . && git commit -m "Fix version consistency"
```

### Manual Override (Emergency)
```bash
# Skip pre-commit hook (not recommended):
git commit --no-verify -m "Emergency fix"
```

## Best Practices

1. **Always use the scripts** - Don't manually edit version numbers
2. **Update VERSION file first** - It's the source of truth
3. **Check consistency** - Run the checker before important commits
4. **Tag releases** - Use `git tag v0.2.17` for releases
5. **Follow semantic versioning** - Be consistent with version increments

This system ensures you'll never forget to update version numbers again! 🎉