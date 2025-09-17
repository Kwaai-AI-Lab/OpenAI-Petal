#!/bin/bash

# Storage Detection and Space Estimation Functions
# Cross-platform compatibility for Linux and macOS

# Function to get available disk space in MB for a given path
get_available_space_mb() {
    local path="$1"
    local space_mb=0

    # Ensure path exists or use parent directory
    while [ ! -d "$path" ] && [ "$path" != "/" ] && [ "$path" != "." ]; do
        path=$(dirname "$path")
    done

    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS - use df with different format
        space_mb=$(df -m "$path" 2>/dev/null | awk 'NR==2 {print $4}')
    else
        # Linux - use df with MB units
        space_mb=$(df -BM "$path" 2>/dev/null | awk 'NR==2 {gsub(/M/, "", $4); print $4}')
    fi

    # Fallback if df fails
    if [ -z "$space_mb" ] || [ "$space_mb" -eq 0 ]; then
        # Try alternative method
        if command -v stat >/dev/null 2>&1; then
            if [[ "$(uname)" == "Darwin" ]]; then
                # macOS stat
                local fs_info=$(stat -f '%a %S' "$path" 2>/dev/null)
                if [ -n "$fs_info" ]; then
                    local blocks=$(echo "$fs_info" | cut -d' ' -f1)
                    local block_size=$(echo "$fs_info" | cut -d' ' -f2)
                    space_mb=$((blocks * block_size / 1024 / 1024))
                fi
            else
                # Linux stat (via statvfs)
                local fs_info=$(stat -f -c '%a %S' "$path" 2>/dev/null)
                if [ -n "$fs_info" ]; then
                    local blocks=$(echo "$fs_info" | cut -d' ' -f1)
                    local block_size=$(echo "$fs_info" | cut -d' ' -f2)
                    space_mb=$((blocks * block_size / 1024 / 1024))
                fi
            fi
        fi
    fi

    echo "${space_mb:-0}"
}

# Function to estimate total installation space requirements
estimate_installation_space() {
    local use_conda="${1:-true}"
    local install_build_tools="${2:-true}"
    local platform="$(uname)"

    local total_mb=0
    local breakdown=""

    # Base system requirements
    local base_mb=100
    total_mb=$((total_mb + base_mb))
    breakdown="Base installation: ${base_mb}MB\n"

    # Git repository
    local repo_mb=50
    total_mb=$((total_mb + repo_mb))
    breakdown="${breakdown}Git repository: ${repo_mb}MB\n"

    # Python environment
    if [ "$use_conda" = "true" ]; then
        # Conda installation
        local conda_mb=500
        total_mb=$((total_mb + conda_mb))
        breakdown="${breakdown}Miniconda: ${conda_mb}MB\n"

        # Conda environment
        local conda_env_mb=200
        total_mb=$((total_mb + conda_env_mb))
        breakdown="${breakdown}Conda environment: ${conda_env_mb}MB\n"
    else
        # Virtual environment
        local venv_mb=50
        total_mb=$((total_mb + venv_mb))
        breakdown="${breakdown}Virtual environment: ${venv_mb}MB\n"
    fi

    # PyTorch (varies by platform and CUDA support)
    local pytorch_mb=2000
    if [[ "$platform" == "Darwin" ]]; then
        # macOS PyTorch is typically smaller
        pytorch_mb=1500
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # CUDA version is larger
        pytorch_mb=3000
    fi
    total_mb=$((total_mb + pytorch_mb))
    breakdown="${breakdown}PyTorch: ${pytorch_mb}MB\n"

    # Other Python packages
    local packages_mb=800
    breakdown="${breakdown}Python packages: ${packages_mb}MB\n"
    breakdown="${breakdown}  - Transformers: 500MB\n"
    breakdown="${breakdown}  - Other dependencies: 300MB\n"
    total_mb=$((total_mb + packages_mb))

    # Build tools (if needed)
    if [ "$install_build_tools" = "true" ]; then
        local build_tools_mb=1000
        if [ "$use_conda" = "true" ]; then
            # Conda build tools are more efficient
            build_tools_mb=300
            breakdown="${breakdown}Build tools (conda): ${build_tools_mb}MB\n"
        else
            breakdown="${breakdown}Build tools (system): ${build_tools_mb}MB\n"
        fi
        total_mb=$((total_mb + build_tools_mb))
    fi

    # Cache and temporary files
    local cache_mb=500
    total_mb=$((total_mb + cache_mb))
    breakdown="${breakdown}Cache/temporary files: ${cache_mb}MB\n"

    # Safety margin (20%)
    local margin_mb=$((total_mb * 20 / 100))
    total_mb=$((total_mb + margin_mb))
    breakdown="${breakdown}Safety margin (20%): ${margin_mb}MB\n"

    # Output results
    echo "TOTAL_MB:$total_mb"
    echo "BREAKDOWN:$breakdown"
}

# Function to check if sufficient space is available
check_storage_requirements() {
    local install_path="${1:-$HOME}"
    local use_conda="${2:-true}"
    local install_build_tools="${3:-true}"

    echo "🔍 Checking storage requirements..."
    echo ""

    # Get space estimation
    local estimation=$(estimate_installation_space "$use_conda" "$install_build_tools")
    local required_mb=$(echo "$estimation" | grep "TOTAL_MB:" | cut -d: -f2)
    local breakdown=$(echo "$estimation" | grep "BREAKDOWN:" | cut -d: -f2-)

    # Get available space
    local available_mb=$(get_available_space_mb "$install_path")

    # Display breakdown
    echo "📊 Space Requirements Breakdown:"
    echo -e "$breakdown"
    echo "----------------------------------------"
    echo "Total required: ${required_mb}MB (~$((required_mb / 1024))GB)"
    echo "Available space: ${available_mb}MB (~$((available_mb / 1024))GB)"
    echo ""

    # Check if we have enough space
    if [ "$available_mb" -lt "$required_mb" ]; then
        local deficit_mb=$((required_mb - available_mb))
        echo "❌ Insufficient storage space!"
        echo "   Need: ${required_mb}MB (~$((required_mb / 1024))GB)"
        echo "   Available: ${available_mb}MB (~$((available_mb / 1024))GB)"
        echo "   Shortfall: ${deficit_mb}MB (~$((deficit_mb / 1024))GB)"
        echo ""
        echo "💡 Suggestions to free up space:"
        echo "   1. Clean up Downloads, Desktop, and temporary files"
        echo "   2. Empty trash/recycle bin"
        echo "   3. Use 'brew cleanup' (macOS) or package manager cleanup"
        echo "   4. Remove old conda environments: 'conda env list' and 'conda env remove -n <name>'"
        echo "   5. Clear pip cache: 'pip cache purge'"
        echo "   6. Clear conda cache: 'conda clean --all'"
        echo ""
        return 1
    else
        local excess_mb=$((available_mb - required_mb))
        echo "✅ Sufficient storage space available!"
        echo "   Required: ${required_mb}MB (~$((required_mb / 1024))GB)"
        echo "   Available: ${available_mb}MB (~$((available_mb / 1024))GB)"
        echo "   Remaining after install: ${excess_mb}MB (~$((excess_mb / 1024))GB)"
        echo ""
        return 0
    fi
}

# Function to provide space optimization suggestions
suggest_space_optimizations() {
    local use_conda="$1"
    local has_cuda="$2"

    echo "💡 Space Optimization Options:"
    echo ""

    if [ "$use_conda" = "true" ]; then
        echo "🔹 Using conda (recommended):"
        echo "   - Saves ~700MB compared to system packages"
        echo "   - More efficient dependency management"
    else
        echo "🔹 Virtual environment mode:"
        echo "   - Minimal overhead (~50MB)"
        echo "   - Requires system build tools (~1GB)"
    fi

    echo ""
    echo "🔹 Build tool options:"
    echo "   - Full build support: ~1GB (can compile any package)"
    echo "   - Pre-built wheels only: ~0MB (use --no-build-tools)"
    echo "   - Conda build tools: ~300MB (recommended compromise)"

    if [ "$has_cuda" = "true" ]; then
        echo ""
        echo "🔹 GPU acceleration:"
        echo "   - CUDA PyTorch: ~3GB (full GPU support)"
        echo "   - CPU-only PyTorch: ~1.5GB (saves ~1.5GB)"
    fi

    echo ""
    echo "🔹 Installation order for limited space:"
    echo "   1. Install without build tools first (--no-build-tools)"
    echo "   2. Test basic functionality"
    echo "   3. Add build tools later if needed"
}

# Function to monitor space during installation
monitor_installation_space() {
    local install_path="${1:-$HOME}"
    local phase="${2:-unknown}"

    local available_mb=$(get_available_space_mb "$install_path")
    local timestamp=$(date '+%H:%M:%S')

    echo "[$timestamp] 📊 $phase - Available space: ${available_mb}MB (~$((available_mb / 1024))GB)"

    # Warn if space is getting low (less than 1GB)
    if [ "$available_mb" -lt 1024 ]; then
        echo "⚠️ Warning: Available space is getting low (${available_mb}MB)"
        echo "   Consider cleaning up temporary files if installation fails"
    fi
}

# Function to get storage info for debugging
get_storage_debug_info() {
    local path="${1:-$HOME}"

    echo "🔍 Storage Debug Information:"
    echo "Target path: $path"
    echo "Platform: $(uname)"
    echo ""

    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Filesystem info (df -h):"
        df -h "$path" 2>/dev/null || echo "df command failed"
        echo ""
        echo "Filesystem info (stat):"
        stat -f '%N: %f free blocks, %S bytes per block' "$path" 2>/dev/null || echo "stat command failed"
    else
        echo "Filesystem info (df -h):"
        df -h "$path" 2>/dev/null || echo "df command failed"
        echo ""
        echo "Filesystem info (stat):"
        stat -f -c '%n: %f free blocks, %S bytes per block' "$path" 2>/dev/null || echo "stat command failed"
    fi

    echo ""
    echo "Available space calculation:"
    local space_mb=$(get_available_space_mb "$path")
    echo "Result: ${space_mb}MB"
}