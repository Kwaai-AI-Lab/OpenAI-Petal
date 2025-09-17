#!/bin/bash

# Error Diagnosis Functions for Package Installation Failures
# Helps distinguish between storage, connectivity, and other issues

# Function to analyze pip install failure and determine root cause
diagnose_pip_failure() {
    local exit_code="$1"
    local error_output="$2"
    local package_name="$3"
    local install_path="${4:-$HOME}"

    echo "🔍 Diagnosing installation failure for $package_name..."

    # Check available space first
    local available_mb=0
    if command -v get_available_space_mb >/dev/null 2>&1; then
        available_mb=$(get_available_space_mb "$install_path")
    fi

    # Analyze error patterns
    local is_storage_issue=false
    local is_connectivity_issue=false
    local is_permission_issue=false
    local is_dependency_issue=false

    # Check for storage-related errors
    if echo "$error_output" | grep -qi -E "(no space|disk full|device full|not enough space|insufficient storage|OSError.*28.*No space|ENOSPC)"; then
        is_storage_issue=true
    elif [ "$available_mb" -lt 1024 ]; then
        # Less than 1GB available - likely storage issue even without explicit error
        is_storage_issue=true
    fi

    # Check for connectivity-related errors
    if echo "$error_output" | grep -qi -E "(connection.*failed|network.*unreachable|timeout|could not resolve|connection.*refused|ssl.*error|certificate.*error|http.*error.*[45][0-9][0-9])"; then
        is_connectivity_issue=true
    fi

    # Check for permission issues
    if echo "$error_output" | grep -qi -E "(permission denied|access denied|operation not permitted|EACCES)"; then
        is_permission_issue=true
    fi

    # Check for dependency conflicts
    if echo "$error_output" | grep -qi -E "(dependency.*conflict|incompatible.*version|requires.*different|version.*conflict)"; then
        is_dependency_issue=true
    fi

    # Provide specific diagnosis
    echo ""
    if [ "$is_storage_issue" = true ]; then
        echo "❌ DIAGNOSIS: Storage space issue detected"
        echo "   Available space: ${available_mb}MB (~$((available_mb / 1024))GB)"
        if [ "$available_mb" -lt 1024 ]; then
            echo "   ⚠️ Critical: Less than 1GB available space"
        fi
        echo ""
        echo "💡 SOLUTIONS:"
        echo "   1. Free up disk space using these commands:"
        echo "      - Clear pip cache: pip cache purge"
        echo "      - Clear conda cache: conda clean --all"
        echo "      - Remove unnecessary files from Downloads, Desktop"
        echo "      - Empty trash/recycle bin"
        echo "   2. Install to a different location with more space"
        echo "   3. Use --no-build-tools flag to save ~1GB"
        return 1
    elif [ "$is_connectivity_issue" = true ]; then
        echo "❌ DIAGNOSIS: Network connectivity issue detected"
        echo "   The download failed due to network problems"
        echo ""
        echo "💡 SOLUTIONS:"
        echo "   1. Check your internet connection"
        echo "   2. Try again in a few minutes (server may be temporarily down)"
        echo "   3. Check if you're behind a firewall or proxy"
        echo "   4. Try using a different network"
        return 2
    elif [ "$is_permission_issue" = true ]; then
        echo "❌ DIAGNOSIS: File permission issue detected"
        echo "   The installer lacks write permissions"
        echo ""
        echo "💡 SOLUTIONS:"
        echo "   1. Run installer with appropriate permissions"
        echo "   2. Check directory permissions: ls -la $install_path"
        echo "   3. Consider using a different installation directory"
        return 3
    elif [ "$is_dependency_issue" = true ]; then
        echo "❌ DIAGNOSIS: Package dependency conflict detected"
        echo "   Version incompatibilities between packages"
        echo ""
        echo "💡 SOLUTIONS:"
        echo "   1. Create a fresh conda environment"
        echo "   2. Update pip: pip install --upgrade pip"
        echo "   3. Clear pip cache: pip cache purge"
        return 4
    else
        echo "❌ DIAGNOSIS: Unknown installation issue"
        echo "   Exit code: $exit_code"
        echo ""
        echo "💡 GENERAL SOLUTIONS:"
        echo "   1. Check available space: ${available_mb}MB (~$((available_mb / 1024))GB)"
        echo "   2. Check internet connectivity"
        echo "   3. Try updating pip: pip install --upgrade pip"
        echo "   4. Clear caches: pip cache purge && conda clean --all"
        echo ""
        echo "📝 Error details:"
        echo "$error_output" | head -10
        return 5
    fi
}

# Function to test connectivity before package installation
test_package_connectivity() {
    local package_name="$1"
    local index_url="$2"

    echo "🌐 Testing connectivity for $package_name..."

    # Test basic internet connectivity
    if ! curl -s --max-time 10 https://pypi.org >/dev/null 2>&1; then
        echo "⚠️ No internet connectivity detected"
        return 1
    fi

    # Test specific index if provided
    if [ -n "$index_url" ]; then
        if ! curl -s --max-time 10 "$index_url" >/dev/null 2>&1; then
            echo "⚠️ Cannot reach package index: $index_url"
            return 2
        fi
    fi

    echo "✅ Package repository connectivity verified"
    return 0
}

# Function to monitor space during package installation
monitor_package_installation() {
    local package_name="$1"
    local install_command="$2"
    local install_path="${3:-$HOME}"

    echo "📦 Installing $package_name with space monitoring..."

    # Get initial space
    local initial_space=0
    if command -v get_available_space_mb >/dev/null 2>&1; then
        initial_space=$(get_available_space_mb "$install_path")
        echo "   Initial space: ${initial_space}MB (~$((initial_space / 1024))GB)"
    fi

    # Check minimum space requirement (1GB for safety)
    if [ "$initial_space" -lt 1024 ] && [ "$initial_space" -gt 0 ]; then
        echo "⚠️ WARNING: Low disk space (${initial_space}MB). Installation may fail."
        echo "   Consider freeing up space before continuing."
    fi

    # Run the installation command and capture output
    local temp_error_file="/tmp/pip_error_$$.log"
    local exit_code=0

    echo "   Running: $install_command"
    if ! eval "$install_command" 2>"$temp_error_file"; then
        exit_code=$?
        local error_output=$(cat "$temp_error_file" 2>/dev/null || echo "No error output captured")

        # Diagnose the failure
        diagnose_pip_failure "$exit_code" "$error_output" "$package_name" "$install_path"
        local diagnosis_code=$?

        # Clean up temp file
        rm -f "$temp_error_file"

        return $diagnosis_code
    else
        # Installation succeeded
        local final_space=0
        if command -v get_available_space_mb >/dev/null 2>&1; then
            final_space=$(get_available_space_mb "$install_path")
            local used_space=$((initial_space - final_space))
            echo "✅ $package_name installed successfully"
            echo "   Space used: ${used_space}MB, remaining: ${final_space}MB (~$((final_space / 1024))GB)"
        else
            echo "✅ $package_name installed successfully"
        fi

        # Clean up temp file
        rm -f "$temp_error_file"

        return 0
    fi
}

# Function to suggest installation strategy based on available space
suggest_installation_strategy() {
    local available_mb="$1"
    local target_package="$2"

    if [ "$available_mb" -lt 2048 ]; then
        echo "💡 Low space detected (${available_mb}MB). Recommended strategy:"
        echo "   1. Use --no-build-tools flag (saves ~1GB)"
        echo "   2. Clear caches first: pip cache purge && conda clean --all"
        echo "   3. Install packages one at a time to monitor space usage"
        if [ "$target_package" = "torch" ]; then
            echo "   4. Consider CPU-only PyTorch version (saves ~1.5GB vs CUDA)"
        fi
    elif [ "$available_mb" -lt 4096 ]; then
        echo "💡 Moderate space available (${available_mb}MB). Recommended strategy:"
        echo "   1. Monitor space during installation"
        echo "   2. Clear caches periodically: pip cache purge"
    else
        echo "✅ Sufficient space available (${available_mb}MB). Standard installation can proceed."
    fi
}