# KwaaiNet for Windows - Uninstaller
# This script completely removes KwaaiNet and its environment from your system

#Requires -Version 5.1

param(
    [switch]$Force,
    [switch]$KeepPython,
    [switch]$KeepCache,
    [switch]$Quiet
)

# Set strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Global variables
$script:InstallPath = "$env:USERPROFILE\.kwaainet"
$script:RemovedItems = @()

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

# Function to write step output
function Write-Step {
    param([string]$Message)
    Write-ColorOutput $Message "Cyan" "==>"
}

# Function to write success output
function Write-Success {
    param([string]$Message)
    Write-ColorOutput $Message "Green" "✅"
}

# Function to write warning output
function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Message "Yellow" "⚠️"
}

# Function to write error output
function Write-ErrorMessage {
    param([string]$Message)
    Write-ColorOutput $Message "Red" "❌"
}

# Function to write info output
function Write-Info {
    param([string]$Message)
    Write-ColorOutput $Message "Blue" "ℹ️"
}

# Function to test if a command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to get user confirmation
function Get-UserConfirmation {
    param(
        [string]$Message,
        [bool]$DefaultYes = $false
    )
    
    if ($Force) {
        return $true
    }
    
    $default = if ($DefaultYes) { "Y" } else { "N" }
    $prompt = "$Message [y/N]"
    if ($DefaultYes) { $prompt = "$Message [Y/n]" }
    
    do {
        $response = Read-Host $prompt
        if ([string]::IsNullOrWhiteSpace($response)) {
            $response = $default
        }
    } while ($response -notin @("Y", "y", "N", "n"))
    
    return $response -in @("Y", "y")
}

# Function to remove launcher scripts
function Remove-LauncherScripts {
    Write-Step "Removing launcher scripts..."
    
    try {
        $launcherDir = "$script:InstallPath\bin"
        
        if (Test-Path $launcherDir) {
            Remove-Item -Path $launcherDir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Success "Removed launcher scripts from $launcherDir"
            $script:RemovedItems += "Launcher scripts"
        }
        else {
            Write-Info "No launcher scripts found"
        }
        
        # Remove from PATH
        try {
            $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
            if ($currentPath -and $currentPath.Contains($launcherDir)) {
                $pathArray = $currentPath -split ";"
                $newPath = ($pathArray | Where-Object { $_ -ne $launcherDir }) -join ";"
                [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
                Write-Success "Removed launcher directory from PATH"
            }
        }
        catch {
            Write-Warning "Failed to update PATH: $($_.Exception.Message)"
        }
    }
    catch {
        Write-Warning "Failed to remove launcher scripts: $($_.Exception.Message)"
    }
}

# Function to remove conda environment
function Remove-CondaEnvironment {
    Write-Step "Checking for kwaainet conda environment..."
    
    try {
        if (Test-Command "conda") {
            # Check if environment exists
            $envList = & conda env list 2>$null | Out-String
            if ($envList -match "kwaainet") {
                if ($KeepPython) {
                    Write-Info "Keeping conda environment (--KeepPython flag specified)"
                    return
                }
                
                if (Get-UserConfirmation "Remove kwaainet conda environment?" $true) {
                    Write-Info "Removing kwaainet conda environment..."
                    
                    # Deactivate any active environment first
                    try { & conda deactivate 2>$null } catch { }
                    
                    $result = & conda env remove -n kwaainet -y 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Conda environment removed successfully"
                        $script:RemovedItems += "Conda environment"
                    }
                    else {
                        Write-Warning "Failed to remove conda environment automatically"
                        Write-Info "You may need to run: conda env remove -n kwaainet -y"
                    }
                }
                else {
                    Write-Info "Keeping conda environment"
                }
            }
            else {
                Write-Info "No kwaainet conda environment found"
            }
        }
        else {
            Write-Info "Conda not found, skipping environment removal"
        }
    }
    catch {
        Write-Warning "Failed to check conda environment: $($_.Exception.Message)"
    }
}

# Function to remove virtual environment
function Remove-VirtualEnvironment {
    Write-Step "Checking for kwaainet virtual environment..."
    
    try {
        $venvPath = "$script:InstallPath\venv"
        
        if (Test-Path $venvPath) {
            if ($KeepPython) {
                Write-Info "Keeping virtual environment (--KeepPython flag specified)"
                return
            }
            
            if (Get-UserConfirmation "Remove kwaainet virtual environment?" $true) {
                Write-Info "Removing kwaainet virtual environment..."
                Remove-Item -Path $venvPath -Recurse -Force
                Write-Success "Virtual environment removed successfully"
                $script:RemovedItems += "Virtual environment"
            }
            else {
                Write-Info "Keeping virtual environment"
            }
        }
        else {
            Write-Info "No virtual environment found"
        }
    }
    catch {
        Write-Warning "Failed to remove virtual environment: $($_.Exception.Message)"
    }
}

# Function to remove Python packages
function Remove-PythonPackages {
    Write-Step "Removing kwaainet packages..."
    
    try {
        $packagesRemoved = $false
        
        # Try to remove from system pip
        if (Test-Command "pip") {
            try {
                $result = & pip uninstall kwaainet-windows kwaainet_windows -y 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Removed kwaainet packages with pip"
                    $packagesRemoved = $true
                }
            }
            catch {
                # Expected if packages not found
            }
        }
        
        if (-not $packagesRemoved) {
            Write-Info "No kwaainet packages found in system Python"
        }
    }
    catch {
        Write-Warning "Failed to remove Python packages: $($_.Exception.Message)"
    }
}

# Function to remove cache directories
function Remove-CacheDirectories {
    if ($KeepCache) {
        Write-Info "Keeping cache directories (--KeepCache flag specified)"
        return
    }
    
    Write-Step "Cleaning up cache directories..."
    
    $cacheDirectories = @(
        "$env:USERPROFILE\.cache\huggingface",
        "$env:USERPROFILE\.cache\transformers",
        "$env:USERPROFILE\.cache\torch",
        "$env:USERPROFILE\.cache\pip",
        "$env:LOCALAPPDATA\pip",
        "$env:LOCALAPPDATA\torch",
        "$env:APPDATA\huggingface"
    )
    
    foreach ($cacheDir in $cacheDirectories) {
        if (Test-Path $cacheDir) {
            if (Get-UserConfirmation "Remove cache directory $cacheDir?" $false) {
                try {
                    Remove-Item -Path $cacheDir -Recurse -Force
                    Write-Success "Removed $cacheDir"
                    $script:RemovedItems += "Cache directory: $(Split-Path $cacheDir -Leaf)"
                }
                catch {
                    Write-Warning "Failed to remove $cacheDir : $($_.Exception.Message)"
                }
            }
            else {
                Write-Info "Skipped $cacheDir"
            }
        }
    }
}

# Function to remove configuration directory
function Remove-ConfigurationDirectory {
    Write-Step "Removing KwaaiNet configuration..."
    
    try {
        if (Test-Path $script:InstallPath) {
            if (Get-UserConfirmation "Remove KwaaiNet configuration directory?" $true) {
                Remove-Item -Path $script:InstallPath -Recurse -Force
                Write-Success "Removed $script:InstallPath"
                $script:RemovedItems += "Configuration directory"
            }
            else {
                Write-Info "Keeping configuration directory"
            }
        }
        else {
            Write-Info "No configuration directory found"
        }
    }
    catch {
        Write-Warning "Failed to remove configuration directory: $($_.Exception.Message)"
    }
}

# Function to clean pip cache
function Clear-PipCache {
    Write-Step "Cleaning pip cache..."
    
    try {
        if (Test-Command "pip") {
            $packages = @("kwaainet-windows", "kwaainet_windows", "petals")
            
            foreach ($package in $packages) {
                try {
                    & pip cache remove $package 2>$null | Out-Null
                }
                catch {
                    # Expected if package cache not found
                }
            }
            Write-Success "Pip cache cleanup completed"
        }
        else {
            Write-Info "Pip not found, skipping cache cleanup"
        }
    }
    catch {
        Write-Warning "Failed to clean pip cache: $($_.Exception.Message)"
    }
}

# Function to stop running processes
function Stop-KwaaineţProcesses {
    Write-Step "Checking for running kwaainet processes..."
    
    try {
        $processes = Get-Process | Where-Object { $_.ProcessName -like "*kwaainet*" -or $_.MainWindowTitle -like "*kwaainet*" }
        
        if ($processes) {
            Write-Info "Found $($processes.Count) running kwaainet process(es)"
            
            if (Get-UserConfirmation "Stop running kwaainet processes?" $true) {
                foreach ($process in $processes) {
                    try {
                        Write-Info "Stopping process: $($process.ProcessName) (PID: $($process.Id))"
                        $process.Kill()
                        $process.WaitForExit(5000)  # Wait up to 5 seconds
                        Write-Success "Stopped process $($process.Id)"
                    }
                    catch {
                        Write-Warning "Failed to stop process $($process.Id): $($_.Exception.Message)"
                    }
                }
                $script:RemovedItems += "Running processes"
            }
            else {
                Write-Info "Keeping running processes"
            }
        }
        else {
            Write-Info "No running kwaainet processes found"
        }
    }
    catch {
        Write-Warning "Failed to check for running processes: $($_.Exception.Message)"
    }
}

# Function to remove Windows services (if any)
function Remove-WindowsServices {
    Write-Step "Checking for kwaainet Windows services..."
    
    try {
        $services = Get-Service | Where-Object { $_.Name -like "*kwaainet*" }
        
        if ($services) {
            Write-Info "Found $($services.Count) kwaainet service(s)"
            
            if (Get-UserConfirmation "Remove kwaainet Windows services?" $true) {
                foreach ($service in $services) {
                    try {
                        Write-Info "Stopping service: $($service.Name)"
                        Stop-Service -Name $service.Name -Force -ErrorAction SilentlyContinue
                        
                        # Remove service (requires admin privileges)
                        if (Test-Path "HKLM:\SYSTEM\CurrentControlSet\Services\$($service.Name)") {
                            Write-Info "Removing service registration: $($service.Name)"
                            & sc.exe delete $service.Name 2>$null
                            Write-Success "Removed service $($service.Name)"
                        }
                    }
                    catch {
                        Write-Warning "Failed to remove service $($service.Name): $($_.Exception.Message)"
                    }
                }
                $script:RemovedItems += "Windows services"
            }
            else {
                Write-Info "Keeping Windows services"
            }
        }
        else {
            Write-Info "No kwaainet services found"
        }
    }
    catch {
        Write-Warning "Failed to check for Windows services: $($_.Exception.Message)"
    }
}

# Function to offer Miniconda removal
function Remove-Miniconda {
    if ($KeepPython) {
        Write-Info "Keeping Miniconda (--KeepPython flag specified)"
        return
    }
    
    Write-Step "Checking for Miniconda installation..."
    
    try {
        $minicondaPath = "$env:USERPROFILE\Miniconda3"
        
        if (Test-Path $minicondaPath) {
            Write-Info "Miniconda installation detected at $minicondaPath"
            Write-Info "This may have been installed by KwaaiNet setup."
            
            if (Get-UserConfirmation "Remove Miniconda installation?" $false) {
                Write-Info "Removing Miniconda..."
                
                # First try to run the uninstaller if it exists
                $uninstaller = "$minicondaPath\Uninstall-Miniconda3.exe"
                if (Test-Path $uninstaller) {
                    try {
                        Write-Info "Running Miniconda uninstaller..."
                        Start-Process $uninstaller -ArgumentList "/S" -Wait -NoNewWindow
                        Write-Success "Miniconda uninstalled successfully"
                    }
                    catch {
                        Write-Warning "Miniconda uninstaller failed, removing manually..."
                        Remove-Item -Path $minicondaPath -Recurse -Force
                        Write-Success "Miniconda removed manually"
                    }
                }
                else {
                    # Manual removal
                    Remove-Item -Path $minicondaPath -Recurse -Force
                    Write-Success "Miniconda removed manually"
                }
                
                # Remove from PATH
                $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
                if ($userPath -and $userPath.Contains("Miniconda3")) {
                    $pathArray = $userPath -split ";"
                    $newPath = ($pathArray | Where-Object { $_ -notlike "*Miniconda3*" }) -join ";"
                    [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
                    Write-Success "Removed Miniconda from PATH"
                }
                
                $script:RemovedItems += "Miniconda installation"
            }
            else {
                Write-Info "Keeping Miniconda installation"
            }
        }
        else {
            Write-Info "No Miniconda installation found"
        }
    }
    catch {
        Write-Warning "Failed to check/remove Miniconda: $($_.Exception.Message)"
    }
}

# Main uninstall function
function Start-Uninstallation {
    Write-Host "==========================================================" -ForegroundColor Red
    Write-Host "KwaaiNet for Windows - Uninstaller" -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
    Write-Host "This will remove KwaaiNet and its environment from your system." -ForegroundColor White
    Write-Host ""
    
    if (-not $Force) {
        if (-not (Get-UserConfirmation "Are you sure you want to uninstall KwaaiNet?" $false)) {
            Write-Info "Uninstallation cancelled."
            return
        }
        Write-Host ""
    }
    
    try {
        # Step 1: Stop running processes
        Stop-KwaaineţProcesses
        
        # Step 2: Remove Windows services
        Remove-WindowsServices
        
        # Step 3: Remove launcher scripts
        Remove-LauncherScripts
        
        # Step 4: Remove conda environment
        Remove-CondaEnvironment
        
        # Step 5: Remove virtual environment
        Remove-VirtualEnvironment
        
        # Step 6: Remove Python packages
        Remove-PythonPackages
        
        # Step 7: Remove cache directories
        Remove-CacheDirectories
        
        # Step 8: Remove configuration directory
        Remove-ConfigurationDirectory
        
        # Step 9: Clean pip cache
        Clear-PipCache
        
        # Step 10: Offer to remove Miniconda
        Remove-Miniconda
        
        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "✅ KwaaiNet has been successfully uninstalled!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Summary of what was removed:" -ForegroundColor White
        if ($script:RemovedItems.Count -gt 0) {
            foreach ($item in $script:RemovedItems) {
                Write-Host "  • $item" -ForegroundColor White
            }
        }
        else {
            Write-Host "  • No components found to remove" -ForegroundColor White
        }
        Write-Host ""
        Write-Host "⚠️ You may need to restart your PowerShell or Command Prompt" -ForegroundColor Yellow
        Write-Host "sessions for PATH changes to take effect." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "📚 Thank you for using KwaaiNet!" -ForegroundColor White
        Write-Host "For support, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal" -ForegroundColor White
        Write-Host "==========================================================" -ForegroundColor Green
    }
    catch {
        Write-ErrorMessage "Uninstallation failed: $($_.Exception.Message)"
        Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
        exit 1
    }
}

# Display help information
function Show-Help {
    Write-Host "KwaaiNet for Windows - Uninstaller" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor White
    Write-Host "  .\windowsuninstaller.ps1 [OPTIONS]" -ForegroundColor Gray
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor White
    Write-Host "  -Force          Skip confirmation prompts and remove everything" -ForegroundColor Gray
    Write-Host "  -KeepPython     Keep Python environments and installations" -ForegroundColor Gray
    Write-Host "  -KeepCache      Keep cache directories" -ForegroundColor Gray
    Write-Host "  -Quiet          Minimize output messages" -ForegroundColor Gray
    Write-Host "  -Help           Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor White
    Write-Host "  .\windowsuninstaller.ps1" -ForegroundColor Gray
    Write-Host "  .\windowsuninstaller.ps1 -Force" -ForegroundColor Gray
    Write-Host "  .\windowsuninstaller.ps1 -KeepPython -KeepCache" -ForegroundColor Gray
    Write-Host ""
}

# Check for help parameter
if ($args -contains "-Help" -or $args -contains "--Help" -or $args -contains "-h" -or $args -contains "/?" -or $args -contains "/h") {
    Show-Help
    exit 0
}

# Script entry point
if ($MyInvocation.InvocationName -ne ".") {
    Start-Uninstallation
}