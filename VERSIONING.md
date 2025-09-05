# Versioning Schema

This project follows [Semantic Versioning (SemVer)](https://semver.org/) for all releases.

## Version Format: MAJOR.MINOR.PATCH

### MAJOR Version (X.0.0)
Increment when making **incompatible API changes** or **breaking changes**:
- Removing or changing existing CLI commands
- Changing configuration file formats that break existing setups
- Removing or changing public API functions
- Changes that require users to modify their existing workflows

**Example**: `0.2.1` → `1.0.0`

### MINOR Version (0.X.0)
Increment when adding **new functionality** in a **backward-compatible** manner:
- Adding new CLI commands or options
- Adding new features to existing commands
- Adding new configuration options (with sensible defaults)
- Performance improvements
- New platform support

**Example**: `0.2.1` → `0.3.0`

### PATCH Version (0.0.X)
Increment when making **backward-compatible bug fixes**:
- Fixing crashes or errors
- Correcting incorrect behavior
- Security patches
- Documentation fixes
- Installation/build fixes

**Example**: `0.2.1` → `0.2.2`

## Implementation

### Files to Update
When bumping versions, update **ALL** of the following files:

#### macOS
- `Installer/macOS/setup.py` - `version="X.Y.Z"`
- `Installer/macOS/kwaainet/__init__.py` - `__version__ = "X.Y.Z"`

#### Linux
- `Installer/linux/setup.py` - `version="X.Y.Z"`
- `Installer/linux/kwaainet/__init__.py` - `__version__ = "X.Y.Z"`

#### Windows
- `Installer/windows/setup.py` - `version="X.Y.Z"`
- `Installer/windows/kwaainet/__init__.py` - `__version__ = "X.Y.Z"`

### Git Workflow
1. Make your changes
2. Update version numbers in all required files
3. Commit changes with descriptive message
4. Tag the release: `git tag v0.2.1`
5. Push with tags: `git push origin main --tags`

## Examples

### Recent Version History
- `v0.2.1` - **PATCH**: Fixed installer temp directory deletion bug and added robust launcher fallback
- `v0.2.0` - **MINOR**: Added daemon mode support with auto-start service
- `v0.1.0` - **MINOR**: Initial release with basic node functionality

### Commit Message Format
```
Brief description of changes

- Detailed change 1
- Detailed change 2
- Version bump: 0.2.0 → 0.2.1 (PATCH: bug fixes)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Pre-Release Versions
For development builds, use pre-release identifiers:
- `0.3.0-alpha.1` - Alpha releases
- `0.3.0-beta.1` - Beta releases  
- `0.3.0-rc.1` - Release candidates

## Version Validation
Before releasing, ensure:
- [ ] All platform setup.py files updated
- [ ] All platform __init__.py files updated
- [ ] Version numbers are consistent across all files
- [ ] CHANGELOG.md updated (if exists)
- [ ] Git tag matches version number