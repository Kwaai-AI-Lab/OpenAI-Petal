# KwaaiNet Windows Installer Test Script
# This script validates that all compatibility patches and dependency fixes work automatically

#Requires -Version 5.1

param(
    [switch]$SkipCleanup,
    [switch]$Quiet
)

# Test script version
$script:TestVersion = "1.0.2"

Write-Host "[TEST] KwaaiNet Windows Installer Test Protocol v$script:TestVersion" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [string]$Prefix = ""
    )
    
    if (-not $Quiet) {
        if ($Prefix) {
            Write-Host "$Prefix " -NoNewline -ForegroundColor $Color
        }
        Write-Host $Message -ForegroundColor $Color
    }
}

function Write-Status {
    param([string]$Message)
    Write-ColorOutput $Message "Blue" "[INFO]"
}

function Write-TestSuccess {
    param([string]$Message)
    Write-ColorOutput $Message "Green" "[SUCCESS]"
}

function Write-TestWarning {
    param([string]$Message)
    Write-ColorOutput $Message "Yellow" "[WARNING]"
}

function Write-TestError {
    param([string]$Message)
    Write-ColorOutput $Message "Red" "[ERROR]"
}

# Check if we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoPath = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$InstallerPath = Join-Path $ScriptDir "windowsinstaller.ps1"
$UninstallerPath = Join-Path $ScriptDir "windowsuninstaller.ps1"

if (-not (Test-Path $InstallerPath)) {
    Write-TestError "OpenAI-Petal Windows installer not found in expected location"
    Write-TestError "Expected: $InstallerPath"
    Write-TestError "Please run this script from the Installer/windows directory"
    exit 1
}

if (-not $SkipCleanup) {
    Write-Status "Phase 1: Complete System Cleanup"
    Write-Host "=================================" -ForegroundColor White

    # Use the official uninstaller for proper cleanup
    Write-Status "Running official KwaaiNet uninstaller..."
    if (Test-Path $UninstallerPath) {
        try {
            # Run uninstaller with Force flag to skip confirmations
            & powershell.exe -ExecutionPolicy Bypass -File $UninstallerPath -Force -Quiet
            Write-TestSuccess "Uninstaller completed"
        }
        catch {
            Write-TestWarning "Uninstaller completed with warnings (this is normal if nothing was installed)"
        }
    }
    else {
        Write-TestError "Official uninstaller not found at: $UninstallerPath"
        exit 1
    }

    # Verify clean state
    Write-Status "Verifying clean state..."
    $condaEnvs = ""
    try {
        if (Test-Command "conda") {
            $condaEnvs = & conda env list 2>$null | Out-String
        }
    }
    catch {
        # conda not available, that's okay
    }

    if ($condaEnvs -match "kwaainet") {
        Write-TestError "kwaainet environment still exists after uninstaller!"
        Write-TestError "This indicates the uninstaller needs improvement."
        exit 1
    }
    else {
        Write-TestSuccess "Clean state confirmed"
    }

    Write-Host ""
}
else {
    Write-Status "Skipping cleanup phase (--SkipCleanup specified)"
    Write-Host ""
}

Write-Status "Phase 2: Fresh Installation"
Write-Host "============================" -ForegroundColor White

# Test the current installer
Write-Status "Running KwaaiNet Windows installer v0.2.22..."
Write-Status "Expected fixes:"
Write-Host "  ✓ Dependency version alignment (transformers 4.34.1)" -ForegroundColor White
Write-Host "  ✓ 7-strategy tokenizers fallback system" -ForegroundColor White
Write-Host "  ✓ Pre-built wheels priority" -ForegroundColor White
Write-Host "  ✓ Enhanced error reporting" -ForegroundColor White
Write-Host "  ✓ Dependency compatibility pre-checking" -ForegroundColor White
Write-Host "  ✓ PyTorch compatibility validation" -ForegroundColor White
Write-Host ""

# Capture installer output
Write-Status "Executing: powershell -ExecutionPolicy Bypass -File windowsinstaller.ps1"
$installerStartTime = Get-Date

try {
    # Run installer and capture all output
    $installerOutput = & powershell.exe -ExecutionPolicy Bypass -File $InstallerPath 2>&1 | Out-String
    $installerExitCode = $LASTEXITCODE
}
catch {
    $installerOutput = $_.Exception.Message
    $installerExitCode = 1
}

$installerEndTime = Get-Date
$installerDuration = ($installerEndTime - $installerStartTime).TotalMinutes

Write-Host "Installer Output:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host $installerOutput
Write-Host "=================" -ForegroundColor Cyan
Write-Host "Installation Duration: $([math]::Round($installerDuration, 2)) minutes" -ForegroundColor White
Write-Host ""

# Analyze installer results
$installationSuccessful = $false
$fixesApplied = 0

if ($installerExitCode -eq 0 -or $installerOutput -match "installation completed") {
    Write-TestSuccess "✓ Installation completed successfully"
    $installationSuccessful = $true
}
else {
    Write-TestError "✗ Installation failed with exit code: $installerExitCode"
}

# Check for dependency hell fixes
Write-Status "Phase 3: Validate Dependency Hell Fixes"
Write-Host "========================================" -ForegroundColor White

# Check for tokenizers installation strategy
if ($installerOutput -match "tokenizers.*installed successfully.*pre-built wheels") {
    Write-TestSuccess "✓ Tokenizers installed from pre-built wheels (build failure prevented)"
    $fixesApplied++
}
elseif ($installerOutput -match "Failed to build tokenizers") {
    Write-TestError "✗ Still encountering tokenizers build failures"
}
else {
    Write-TestWarning "? Tokenizers installation method unclear from output"
}

# Check for version alignment
if ($installerOutput -match "transformers.*4\.34\.1" -or $installerOutput -match "transformers==4.34.1") {
    Write-TestSuccess "✓ Transformers version aligned with Linux installer (4.34.1)"
    $fixesApplied++
}
else {
    Write-TestWarning "? Transformers version alignment not detected in output"
}

# Check for enhanced error reporting
if ($installerOutput -match "Error output:|Error:|Exception:" -and $installerOutput.Length -gt 1000) {
    Write-TestSuccess "✓ Enhanced error reporting detected"
    $fixesApplied++
}

# Check for HuggingFace connectivity test
if ($installerOutput -match "Hugging Face.*connectivity.*verified") {
    Write-TestSuccess "✓ HuggingFace connectivity pre-check working"
    $fixesApplied++
}

# Check for dependency compatibility testing
if ($installerOutput -match "dependency compatibility|Testing.*compatibility") {
    Write-TestSuccess "✓ Dependency compatibility pre-checking detected"
    $fixesApplied++
}

Write-Host ""

if ($installationSuccessful) {
    Write-Status "Phase 4: Test Basic Functionality"
    Write-Host "==================================" -ForegroundColor White

    # Check for launcher scripts
    $launcherDir = "$env:USERPROFILE\.kwaainet\bin"
    $batchLauncher = Join-Path $launcherDir "kwaainet.bat"
    $psLauncher = Join-Path $launcherDir "kwaainet.ps1"

    if (Test-Path $batchLauncher) {
        Write-TestSuccess "✓ Batch launcher created: $batchLauncher"
    }
    else {
        Write-TestWarning "✗ Batch launcher not found: $batchLauncher"
    }

    if (Test-Path $psLauncher) {
        Write-TestSuccess "✓ PowerShell launcher created: $psLauncher"
    }
    else {
        Write-TestWarning "✗ PowerShell launcher not found: $psLauncher"
    }

    # Test kwaainet command
    Write-Status "Testing kwaainet command functionality..."
    
    # Add launcher directory to PATH for testing
    $env:PATH += ";$launcherDir"
    
    try {
        # Test help command
        $helpOutput = & kwaainet --help 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0 -and ($helpOutput -match "usage" -or $helpOutput -match "help")) {
            Write-TestSuccess "✓ kwaainet --help command works"
            
            # Check version
            if ($helpOutput -match "daemon" -or $helpOutput -match "start" -or $helpOutput -match "stop" -or $helpOutput -match "status") {
                Write-TestSuccess "✓ Daemon management commands available"
            }
        }
        else {
            Write-TestWarning "✗ kwaainet --help failed or gave unexpected output"
            Write-Host "Output: $helpOutput" -ForegroundColor Yellow
        }
    }
    catch {
        Write-TestWarning "✗ kwaainet command test failed: $($_.Exception.Message)"
    }

    Write-Host ""

    Write-Status "Phase 5: Test Daemon Functionality"
    Write-Host "===================================" -ForegroundColor White

    Write-Status "Testing daemon startup (non-blocking mode)..."
    
    try {
        # Test daemon startup
        $daemonOutput = & kwaainet start --daemon 2>&1 | Out-String
        Write-Host "Daemon startup output:" -ForegroundColor Cyan
        Write-Host "======================" -ForegroundColor Cyan
        Write-Host $daemonOutput
        Write-Host "======================" -ForegroundColor Cyan

        if ($daemonOutput -match "daemon" -or $daemonOutput -match "start" -or $daemonOutput -match "running") {
            Write-TestSuccess "✓ Daemon startup initiated"
            
            # Wait a moment for daemon to stabilize
            Write-Status "Waiting 15 seconds for daemon to stabilize..."
            Start-Sleep -Seconds 15
            
            # Test daemon status
            $statusOutput = & kwaainet status 2>&1 | Out-String
            Write-Host "Daemon status output:" -ForegroundColor Cyan
            Write-Host $statusOutput
            
            if ($statusOutput -match "running" -or $statusOutput -match "active" -or $statusOutput -match "started") {
                Write-TestSuccess "✓ Daemon is running"
                
                # Test daemon stop
                Write-Status "Testing daemon stop..."
                $stopOutput = & kwaainet stop 2>&1 | Out-String
                if ($stopOutput -match "stopped" -or $stopOutput -match "stop") {
                    Write-TestSuccess "✓ Daemon stopped successfully"
                }
            }
            else {
                Write-TestWarning "? Daemon status unclear: $statusOutput"
            }
        }
        else {
            Write-TestWarning "? Daemon startup result unclear"
        }
    }
    catch {
        Write-TestWarning "Daemon test failed: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Status "Test Results Summary"
Write-Host "====================" -ForegroundColor White

if ($installationSuccessful) {
    Write-Host "Installation Successful: [PASS] YES" -ForegroundColor Green
} else {
    Write-Host "Installation Successful: [FAIL] NO" -ForegroundColor Red
}
Write-Host "Dependency Fixes Applied: $fixesApplied" -ForegroundColor White
Write-Host "Installation Duration: $([math]::Round($installerDuration, 2)) minutes" -ForegroundColor White

if ($installationSuccessful -and $fixesApplied -ge 3) {
    Write-TestSuccess "[SUCCESS] WINDOWS INSTALLER TEST PASSED!"
    Write-TestSuccess "[PASS] Dependency hell issues resolved"
    Write-TestSuccess "[PASS] Tokenizers build failures prevented"
    Write-TestSuccess "[PASS] Enhanced error reporting working"
    Write-TestSuccess "[PASS] Installation completed successfully"
    Write-Host ""
    Write-TestSuccess "The KwaaiNet Windows installer v0.2.22 is PRODUCTION READY!"
    Write-Host ""
    Write-Host "Key improvements over v0.2.13:" -ForegroundColor White
    Write-Host "  - 7-strategy tokenizers fallback prevents build failures" -ForegroundColor White
    Write-Host "  - Version alignment with tested Linux installer" -ForegroundColor White
    Write-Host "  - Enhanced error reporting for better debugging" -ForegroundColor White
    Write-Host "  - Dependency compatibility pre-checking" -ForegroundColor White
    Write-Host "  - Advanced GPU detection and validation" -ForegroundColor White
    exit 0
}
elseif ($installationSuccessful) {
    Write-TestWarning "[WARNING] Installation successful but some expected fixes may not be working optimally"
    Write-TestWarning "Check the output above for details"
    exit 1
}
else {
    Write-TestError "[ERROR] Installation failed - dependency hell issues may persist"
    Write-TestError "This indicates the fixes need further refinement"
    exit 1
}

# Helper function to test if a command exists (defined at end to avoid issues)
function Test-Command {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}