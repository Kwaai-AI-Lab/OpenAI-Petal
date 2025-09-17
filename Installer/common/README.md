# Common Installer Functions

This directory contains shared functions used by both Linux and macOS installers.

## Storage Check Functions

### `storage_check.sh`

Provides comprehensive storage detection and space estimation functionality:

**Key Functions:**
- `get_available_space_mb(path)` - Cross-platform disk space detection
- `estimate_installation_space(use_conda, install_build_tools)` - Calculate space requirements
- `check_storage_requirements(install_path, use_conda, install_build_tools)` - Pre-flight storage check
- `monitor_installation_space(install_path, phase)` - Real-time space monitoring
- `suggest_space_optimizations(use_conda, has_cuda)` - Space saving recommendations

**Space Estimates:**
- **Linux with conda + build tools**: ~4.7GB
- **macOS with conda (no build tools)**: ~4.4GB
- **Linux with venv + build tools**: ~4.8GB

**Features:**
- Cross-platform compatibility (Linux/macOS)
- Detailed breakdown of space requirements
- Pre-installation validation with helpful error messages
- Real-time monitoring during installation phases
- Space optimization suggestions
- Configurable via environment variables

**Usage in Installers:**
```bash
# Source the functions
source "$(dirname "$0")/../common/storage_check.sh"

# Check requirements before installation
if ! check_storage_requirements "$HOME" "true" "true"; then
    exit 1
fi

# Monitor during installation
monitor_installation_space "$HOME" "After PyTorch installation"
```

**Environment Variables:**
- `SKIP_STORAGE_CHECK=true` - Skip storage validation
- `KWAAINET_STORAGE_DEBUG=true` - Enable debug output