#!/bin/bash

# Common Installer Functions
# Shared utilities for Linux and macOS installers
# Version: 0.6.0

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

# Log success message with checkmark emoji
log_success() {
    echo "✅ $*"
}

# Log error message with X emoji
log_error() {
    echo "❌ $*" >&2
}

# Log warning message with warning emoji
log_warning() {
    echo "⚠️  $*"
}

# Log info message with package emoji
log_info() {
    echo "📦 $*"
}

# Log step/section header
log_step() {
    echo ""
    echo "📦 $*"
}

# Log search/detection message
log_search() {
    echo "🔍 $*"
}

# ============================================================================
# ERROR HANDLING
# ============================================================================

# Fatal error - log and exit
die() {
    log_error "$*"
    exit 1
}

# Warning - log and continue
warn_continue() {
    log_warning "$*"
}

# ============================================================================
# COMMAND UTILITIES
# ============================================================================

# Check if command exists (silent)
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Require a command or exit with error
require_command() {
    local cmd="$1"
    local pkg="${2:-$1}"

    if ! command_exists "$cmd"; then
        die "$cmd is required but not found. Please install: $pkg"
    fi
}

# Find best available Python command
find_python_cmd() {
    local python_cmd=""

    # Try in order of preference
    for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command_exists "$cmd"; then
            # Verify it's Python 3.8+
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major=$(echo "$version" | cut -d. -f1)
            local minor=$(echo "$version" | cut -d. -f2)

            if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
                python_cmd="$cmd"
                break
            fi
        fi
    done

    echo "$python_cmd"
}

# Test if Python module can be imported
test_python_module() {
    local module="$1"
    local python_cmd="${2:-python3}"

    if [[ -n "$python_cmd" ]] && command_exists "$python_cmd"; then
        $python_cmd -c "import $module" >/dev/null 2>&1
        return $?
    fi
    return 1
}

# Get Python module version
get_module_version() {
    local module="$1"
    local python_cmd="${2:-python3}"

    if test_python_module "$module" "$python_cmd"; then
        $python_cmd -c "import $module; print($module.__version__)" 2>/dev/null
    fi
}

# ============================================================================
# PACKAGE MANAGEMENT
# ============================================================================

# Install single pip package with error handling
install_pip_package() {
    local package="$1"
    local pip_cmd="${2:-pip}"
    local extra_args="${3:-}"

    log_info "Installing $package..."

    if $pip_cmd install $extra_args "$package"; then
        log_success "$package installed successfully"
        return 0
    else
        log_error "Failed to install $package"
        return 1
    fi
}

# Install multiple pip packages in batch
install_pip_packages_batch() {
    local packages="$1"
    local pip_cmd="${2:-pip}"
    local extra_args="${3:-}"

    log_info "Installing packages: $packages"

    if $pip_cmd install $extra_args $packages; then
        log_success "Packages installed successfully"
        return 0
    else
        log_error "Failed to install packages"
        return 1
    fi
}

# Verify package is installed
verify_package_installed() {
    local package="$1"
    local pip_cmd="${2:-pip}"

    $pip_cmd show "$package" >/dev/null 2>&1
}

# Get installed package version
get_package_version() {
    local package="$1"
    local pip_cmd="${2:-pip}"

    $pip_cmd show "$package" 2>/dev/null | grep "^Version:" | awk '{print $2}'
}

# ============================================================================
# CONDA UTILITIES
# ============================================================================

# Check if conda is available
conda_available() {
    command_exists conda || [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]
}

# Source conda initialization
source_conda() {
    local conda_sh="${1:-$HOME/miniconda3/etc/profile.d/conda.sh}"

    if [ -f "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        return 0
    fi
    return 1
}

# Create conda environment
create_conda_env() {
    local env_name="$1"
    local python_version="${2:-3.10}"

    log_info "Creating conda environment: $env_name (Python $python_version)"

    if conda create -n "$env_name" python="$python_version" -y; then
        log_success "Conda environment created: $env_name"
        return 0
    else
        log_error "Failed to create conda environment: $env_name"
        return 1
    fi
}

# Activate conda environment
activate_conda_env() {
    local env_name="$1"

    if conda_available; then
        source_conda
        conda activate "$env_name" 2>/dev/null || {
            log_error "Failed to activate conda environment: $env_name"
            return 1
        }
        log_success "Activated conda environment: $env_name"
        return 0
    fi
    return 1
}

# Check if conda environment exists
conda_env_exists() {
    local env_name="$1"

    if conda_available; then
        conda env list | grep -q "^$env_name "
        return $?
    fi
    return 1
}

# ============================================================================
# SHELL CONFIGURATION
# ============================================================================

# Add line to file if it doesn't exist
# Usage: add_line_if_not_exists FILE LINE [COMMENT]
add_line_if_not_exists() {
    local file="$1"
    local line="$2"
    local comment="${3:-}"

    if [ ! -f "$file" ]; then
        touch "$file"
    fi

    # Escape the line for grep
    local escaped_line=$(echo "$line" | sed 's/[]\/$*.^|[]/\\&/g')

    if ! grep -q "$escaped_line" "$file"; then
        if [ -n "$comment" ]; then
            echo "" >> "$file"
            echo "$comment" >> "$file"
        fi
        echo "$line" >> "$file"
        log_success "Added to $file"
        return 0
    fi
    return 1
}

# Setup shell RC file with content
setup_shell_rc() {
    local content="$1"
    local shell_name="${2:-bash}"

    local rc_file=""
    case "$shell_name" in
        bash)
            rc_file="$HOME/.bashrc"
            ;;
        zsh)
            rc_file="$HOME/.zshrc"
            ;;
        *)
            log_warning "Unknown shell: $shell_name"
            return 1
            ;;
    esac

    add_line_if_not_exists "$content" "$rc_file"
}

# Detect user's default shell
detect_shell() {
    local shell_path="${SHELL:-/bin/bash}"
    basename "$shell_path"
}

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

# Detect OS type
detect_os() {
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "macos"
    elif [[ "$(uname)" == "Linux" ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# Detect Linux distribution family
detect_distro_family() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release

        case "$ID_LIKE" in
            *debian*|*ubuntu*)
                echo "debian"
                ;;
            *rhel*|*fedora*|*centos*)
                echo "rhel"
                ;;
            *suse*)
                echo "suse"
                ;;
            *arch*)
                echo "arch"
                ;;
            *)
                # Fallback to ID if ID_LIKE not set
                case "$ID" in
                    debian|ubuntu)
                        echo "debian"
                        ;;
                    rhel|fedora|centos|rocky|alma)
                        echo "rhel"
                        ;;
                    opensuse*|sles)
                        echo "suse"
                        ;;
                    arch|manjaro)
                        echo "arch"
                        ;;
                    *)
                        echo "unknown"
                        ;;
                esac
                ;;
        esac
    else
        echo "unknown"
    fi
}

# Check if running as root
is_root() {
    [ "$(id -u)" -eq 0 ]
}

# Check if sudo is available
has_sudo() {
    command_exists sudo && sudo -n true 2>/dev/null
}

# ============================================================================
# FILE UTILITIES
# ============================================================================

# Create directory if it doesn't exist
ensure_directory() {
    local dir="$1"

    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || die "Failed to create directory: $dir"
    fi
}

# Backup file if it exists
backup_file() {
    local file="$1"
    local backup_suffix="${2:-.bak}"

    if [ -f "$file" ]; then
        local backup="${file}${backup_suffix}.$(date +%Y%m%d_%H%M%S)"
        cp "$file" "$backup"
        log_success "Backed up $file to $backup"
    fi
}

# ============================================================================
# VERSION COMPARISON
# ============================================================================

# Compare two version strings (returns 0 if v1 >= v2)
version_ge() {
    local v1="$1"
    local v2="$2"

    # Use sort -V for version comparison
    if printf '%s\n%s\n' "$v2" "$v1" | sort -V -C 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# SPINNER/PROGRESS
# ============================================================================

# Show spinner while command runs
show_spinner() {
    local pid=$1
    local message="${2:-Processing...}"
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    while ps -p "$pid" > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c] %s" "$spinstr" "$message"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\r"
    done
    printf "    \r"
}

# ============================================================================
# CLEANUP
# ============================================================================

# Register cleanup function to run on exit
cleanup_on_exit() {
    local cleanup_func="$1"
    trap "$cleanup_func" EXIT INT TERM
}

# ============================================================================
# EXPORTS
# ============================================================================

# Export all functions for use in sourcing scripts
export -f log_success log_error log_warning log_info log_step log_search
export -f die warn_continue
export -f command_exists require_command find_python_cmd test_python_module get_module_version
export -f install_pip_package install_pip_packages_batch verify_package_installed get_package_version
export -f conda_available source_conda create_conda_env activate_conda_env conda_env_exists
export -f add_line_if_not_exists setup_shell_rc detect_shell
export -f detect_os detect_distro_family is_root has_sudo
export -f ensure_directory backup_file
export -f version_ge
export -f show_spinner cleanup_on_exit
