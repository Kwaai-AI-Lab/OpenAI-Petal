# KwaaiNet for Windows - One-Step Installer v0.2.10
# This script handles the entire installation process for KwaaiNet on Windows

# Ensure we can run PowerShell scripts
#Requires -Version 5.1

param(
    [switch]$UseSystemPython,
    [switch]$UseConda,
    [switch]$Force,
    [switch]$Quiet
)

# Installer version
$script:InstallerVersion = "0.2.10"

# Output version immediately for debugging
Write-Host "KwaaiNet Windows Installer v$script:InstallerVersion starting..." -ForegroundColor Green

# Set strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Global variables
$script:InstallPath = Join-Path $env:USERPROFILE ".kwaainet"
$script:PythonMethod = ""
$script:GpuType = "none"
$script:GpuInfo = ""
$script:UseElevated = $false
$script:Quiet = $Quiet.IsPresent
$script:UseConda = $UseConda.IsPresent
$script:UseSystemPython = $UseSystemPython.IsPresent
$script:Force = $Force.IsPresent

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [string]$Prefix = ""
    )
    
    if (-not $script:Quiet) {
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
    Write-ColorOutput $Message "Green" "[SUCCESS]"
}

# Function to write warning output
function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Message "Yellow" "[WARNING]"
}

# Function to write error output
function Write-ErrorMessage {
    param([string]$Message)
    Write-ColorOutput $Message "Red" "[ERROR]"
}

# Function to write info output
function Write-Info {
    param([string]$Message)
    Write-ColorOutput $Message "Blue" "[INFO]"
}

# Function to test if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to detect Windows version and architecture
function Get-SystemInfo {
    Write-Step "Detecting Windows version and architecture..."
    
    try {
        $osInfo = Get-CimInstance Win32_OperatingSystem
        $computerInfo = Get-CimInstance Win32_ComputerSystem
        
        $osVersion = $osInfo.Version
        $osName = $osInfo.Caption
        $architecture = $computerInfo.SystemType
        $totalRAM = [math]::Round($computerInfo.TotalPhysicalMemory / 1GB, 2)
        
        Write-Success "Windows System Detected:"
        Write-Host "   OS: $osName" -ForegroundColor White
        Write-Host "   Version: $osVersion" -ForegroundColor White  
        Write-Host "   Architecture: $architecture" -ForegroundColor White
        Write-Host "   Total RAM: ${totalRAM} GB" -ForegroundColor White
        
        # Check Windows 10/11 compatibility
        $majorVersion = [int]($osVersion.Split('.')[0])
        $buildNumber = [int]($osVersion.Split('.')[2])
        
        if ($majorVersion -lt 10) {
            Write-ErrorMessage "Windows 10 or newer is required. Found Windows version $majorVersion."
            exit 1
        }
        
        # Check architecture
        if ($architecture -notmatch "x64" -and $architecture -notmatch "ARM64") {
            Write-ErrorMessage "64-bit Windows is required. Found: $architecture"
            exit 1
        }
        
        return @{
            OSName = $osName
            OSVersion = $osVersion
            Architecture = $architecture
            TotalRAM = $totalRAM
            BuildNumber = $buildNumber
        }
    }
    catch {
        Write-ErrorMessage "Failed to detect system information: $($_.Exception.Message)"
        exit 1
    }
}

# Function to check PowerShell execution policy
function Test-ExecutionPolicy {
    Write-Step "Checking PowerShell execution policy..."
    
    $currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
    $validPolicies = @("Unrestricted", "RemoteSigned", "Bypass")
    
    if ($currentPolicy -notin $validPolicies) {
        Write-Warning "Current execution policy ($currentPolicy) may prevent installation."
        
        try {
            Write-Info "Attempting to set execution policy for current user..."
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Write-Success "Execution policy updated to RemoteSigned for current user"
        }
        catch {
            Write-ErrorMessage "Failed to update execution policy. Please run:"
            Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
            exit 1
        }
    }
    else {
        Write-Success "Execution policy is compatible: $currentPolicy"
    }
}

# Function to detect GPU hardware
function Get-GpuInfo {
    Write-Step "Detecting GPU hardware..."
    
    try {
        $script:GpuType = "none"
        $script:GpuInfo = ""
        
        # Check for NVIDIA GPU first
        try {
            $nvidiaOutput = & nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
            if ($LASTEXITCODE -eq 0 -and $nvidiaOutput) {
                $script:GpuType = "nvidia"
                $script:GpuInfo = $nvidiaOutput.Trim()
                Write-Success "NVIDIA GPU detected: $script:GpuInfo"
                return
            }
        }
        catch {
            # nvidia-smi not available, continue checking
        }
        
        # Check GPU via WMI
        $gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notlike "*Basic*" -and $_.Name -notlike "*Generic*" }
        
        foreach ($gpu in $gpus) {
            $gpuName = $gpu.Name
            
            if ($gpuName -match "NVIDIA|GeForce|GTX|RTX|Quadro|Tesla") {
                $script:GpuType = "nvidia"
                $script:GpuInfo = $gpuName
                Write-Success "NVIDIA GPU detected: $gpuName"
                Write-Warning "NVIDIA drivers may not be properly installed (nvidia-smi not found)"
                break
            }
            elseif ($gpuName -match "AMD|Radeon|RX|Vega") {
                $script:GpuType = "amd"
                $script:GpuInfo = $gpuName
                Write-Success "AMD GPU detected: $gpuName"
                break
            }
            elseif ($gpuName -match "Intel.*Graphics|Intel.*Iris|Intel.*HD") {
                $script:GpuType = "intel"
                $script:GpuInfo = $gpuName
                Write-Success "Intel GPU detected: $gpuName"
            }
        }
        
        if ($script:GpuType -eq "none") {
            Write-Info "No dedicated GPU detected. Using CPU-only mode."
        }
    }
    catch {
        Write-Warning "Failed to detect GPU information: $($_.Exception.Message)"
        $script:GpuType = "none"
    }
}

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to install package via winget
function Install-WingetPackage {
    param(
        [string]$PackageId,
        [string]$DisplayName,
        [switch]$Silent = $true
    )
    
    if (-not (Test-Command "winget")) {
        Write-Warning "winget not available. Please install $DisplayName manually."
        return $false
    }
    
    try {
        $args = @("install", "--id", $PackageId, "--accept-package-agreements", "--accept-source-agreements")
        if ($Silent) { $args += "--silent" }
        
        Write-Info "Installing $DisplayName via winget..."
        $process = Start-Process "winget" -ArgumentList $args -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -eq 0) {
            Write-Success "$DisplayName installed successfully"
            return $true
        }
        else {
            Write-Warning "winget installation of $DisplayName failed (exit code: $($process.ExitCode))"
            return $false
        }
    }
    catch {
        Write-Warning "Failed to install $DisplayName via winget: $($_.Exception.Message)"
        return $false
    }
}

# Function to detect and setup Python environment
function Set-PythonEnvironment {
    Write-Step "Setting up Python environment..."
    
    try {
        # Determine which Python method to use
        if ($script:UseConda) {
            $script:PythonMethod = "conda"
            Write-Info "Using conda (forced by --UseConda flag)"
        }
        elseif ($script:UseSystemPython) {
            $script:PythonMethod = "system"
            Write-Info "Using system Python (forced by --UseSystemPython flag)"
        }
        else {
            # Auto-detect best method
            if (Test-Command "conda") {
                $script:PythonMethod = "conda"
                Write-Success "Using existing conda installation"
            }
            elseif (Test-Command "python") {
                try {
                    $pythonVersion = (& python --version 2>&1).ToString()
                    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
                        $major = [int]$matches[1]
                        $minor = [int]$matches[2]
                        
                        if ($major -ge 3 -and $minor -ge 8) {
                            $script:PythonMethod = "system"
                            Write-Success "Using system Python $($matches[0])"
                        }
                        else {
                            Write-Warning "System Python is too old ($($matches[0])). Will install conda..."
                            $script:PythonMethod = "conda"
                        }
                    }
                    else {
                        Write-Warning "Could not determine Python version. Will install conda..."
                        $script:PythonMethod = "conda"
                    }
                }
                catch {
                    Write-Warning "Python found but version check failed. Will install conda..."
                    $script:PythonMethod = "conda"
                }
            }
            else {
                Write-Warning "No suitable Python found. Will install conda..."
                $script:PythonMethod = "conda"
            }
        }
    
        # Install or setup the chosen Python environment
        if ($script:PythonMethod -eq "conda") {
            Install-Conda
            Setup-CondaEnvironment
        }
        else {
            Setup-SystemPython
        }
        
        Write-Success "Python environment setup completed successfully"
    }
    catch {
        Write-ErrorMessage "Failed to setup Python environment: $($_.Exception.Message)"
        Write-Host "Error details: $($_.Exception)" -ForegroundColor Red
        exit 1
    }
}

# Function to install Miniconda
function Install-Conda {
    Write-Step "Installing Miniconda..."
    
    # Check if conda is already available
    if (Test-Command "conda") {
        Write-Success "Conda already installed"
        return
    }
    
    try {
        # Determine architecture and download URL
        $arch = if ([Environment]::Is64BitOperatingSystem) { "x86_64" } else { "x86" }
        $installerName = "Miniconda3-latest-Windows-$arch.exe"
        $downloadUrl = "https://repo.anaconda.com/miniconda/$installerName"
        $installerPath = "$env:TEMP\$installerName"
        
        Write-Info "Downloading Miniconda installer..."
        try {
            # Use Invoke-WebRequest with proper headers to avoid 403 errors
            $progressPreference = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'  # Suppress progress bar for faster download
            
            Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UserAgent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -UseBasicParsing
            
            $ProgressPreference = $progressPreference  # Restore original setting
        }
        catch {
            Write-Warning "Failed to download with Invoke-WebRequest, trying WebClient with headers..."
            try {
                $webClient = New-Object System.Net.WebClient
                $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                $webClient.DownloadFile($downloadUrl, $installerPath)
            }
            catch {
                Write-ErrorMessage "Failed to download Miniconda installer: $($_.Exception.Message)"
                Write-Info "Please try downloading manually from: $downloadUrl"
                throw
            }
        }
        
        # Verify download was successful
        if (-not (Test-Path $installerPath) -or (Get-Item $installerPath).Length -eq 0) {
            Write-ErrorMessage "Miniconda installer download failed or file is empty"
            Write-Info "Please check your internet connection and try again"
            exit 1
        }
        
        Write-Success "Miniconda installer downloaded successfully ($('{0:N2}' -f ((Get-Item $installerPath).Length / 1MB)) MB)"
        Write-Info "Installing Miniconda (this may take a few minutes)..."
        $installArgs = @(
            "/S",  # Silent install
            "/InstallationType=JustMe",
            "/AddToPath=1",
            "/RegisterPython=0",
            "/D=$env:USERPROFILE\Miniconda3"
        )
        
        $process = Start-Process $installerPath -ArgumentList $installArgs -Wait -NoNewWindow -PassThru
        
        if ($process.ExitCode -ne 0) {
            Write-ErrorMessage "Miniconda installation failed with exit code $($process.ExitCode)"
            exit 1
        }
        
        # Clean up installer
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        
        # Update PATH for current session
        $condaPath = "$env:USERPROFILE\Miniconda3"
        $env:PATH = "$condaPath;$condaPath\Scripts;$condaPath\Library\bin;$env:PATH"
        
        Write-Success "Miniconda installed successfully"
    }
    catch {
        Write-ErrorMessage "Failed to install Miniconda: $($_.Exception.Message)"
        exit 1
    }
}

# Function to setup conda environment
function Setup-CondaEnvironment {
    Write-Step "Setting up KwaaiNet conda environment..."
    
    # Ensure conda is properly available and PATH is set
    $condaPath = "$env:USERPROFILE\Miniconda3"
    if (Test-Path "$condaPath\Scripts\conda.exe") {
        # Add conda to PATH for current session
        $env:PATH = "$condaPath\Scripts;$condaPath;$condaPath\Library\bin;$env:PATH"
        Write-Info "Updated PATH for conda access: $condaPath"
    }
    else {
        Write-ErrorMessage "Conda executable not found at: $condaPath\Scripts\conda.exe"
        Write-Info "Please ensure Miniconda installed successfully"
        exit 1
    }
    
    # Test conda command availability with direct path
    $condaExe = "$condaPath\Scripts\conda.exe"
    if (-not (Test-Path $condaExe)) {
        Write-ErrorMessage "Conda executable missing: $condaExe"
        exit 1
    }
    
    # Check if environment already exists using direct conda path
    Write-Info "Checking for existing conda environments..."
    try {
        $envList = & $condaExe env list 2>$null
        if ($LASTEXITCODE -eq 0 -and $envList -match "kwaainet") {
            Write-Success "Using existing kwaainet conda environment"
            return
        }
    }
    catch {
        Write-Warning "Could not check existing environments, will attempt to create new one"
    }
    
    # Create new environment
    Write-Info "Creating Python 3.10 environment for KwaaiNet..."
    Write-Info "This may take several minutes to download packages..."
    
    try {
        # Test conda executable directly
        Write-Info "Testing conda installation with direct executable path..."
        $condaVersion = & $condaExe --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ErrorMessage "Conda executable failed to run. Exit code: $LASTEXITCODE"
            Write-Info "Conda path: $condaExe"
            exit 1
        }
        Write-Info "Conda version: $condaVersion"
        
        # Configure conda channels to avoid Terms of Service issues BEFORE creating environment
        Write-Info "Configuring conda channels to avoid Terms of Service issues..."
        try {
            # Remove problematic Anaconda commercial channels
            $channelRemoveArgs = @("config", "--remove", "channels", "https://repo.anaconda.com/pkgs/main")
            Start-Process -FilePath $condaExe -ArgumentList $channelRemoveArgs -Wait -NoNewWindow -PassThru | Out-Null
            
            $channelRemoveArgs = @("config", "--remove", "channels", "https://repo.anaconda.com/pkgs/r")  
            Start-Process -FilePath $condaExe -ArgumentList $channelRemoveArgs -Wait -NoNewWindow -PassThru | Out-Null
            
            $channelRemoveArgs = @("config", "--remove", "channels", "https://repo.anaconda.com/pkgs/msys2")
            Start-Process -FilePath $condaExe -ArgumentList $channelRemoveArgs -Wait -NoNewWindow -PassThru | Out-Null
            
            # Add conda-forge as the primary channel
            $channelAddArgs = @("config", "--add", "channels", "conda-forge")
            Start-Process -FilePath $condaExe -ArgumentList $channelAddArgs -Wait -NoNewWindow -PassThru | Out-Null
            
            # Set channel priority to strict
            $channelPriorityArgs = @("config", "--set", "channel_priority", "strict")
            Start-Process -FilePath $condaExe -ArgumentList $channelPriorityArgs -Wait -NoNewWindow -PassThru | Out-Null
            
            Write-Success "Configured conda to use conda-forge channel (avoids Terms of Service issues)"
        }
        catch {
            Write-Warning "Could not configure conda channels, but continuing installation..."
        }
        
        # Create environment using direct executable path to avoid PowerShell execution context issues
        Write-Info "Creating conda environment (this may take 5-10 minutes)..."
        
        # Use Start-Process instead of & operator to avoid RemoteException
        # Explicitly use conda-forge channel to avoid Terms of Service issues
        $processArgs = @("create", "-y", "-n", "kwaainet", "python=3.10", "-c", "conda-forge", "--override-channels")
        Write-Info "Running: $condaExe $($processArgs -join ' ')"
        
        $process = Start-Process -FilePath $condaExe -ArgumentList $processArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\conda_out.txt" -RedirectStandardError "$env:TEMP\conda_err.txt"
        $condaExitCode = $process.ExitCode
        
        # Read output files
        $condaOutput = ""
        $condaError = ""
        if (Test-Path "$env:TEMP\conda_out.txt") {
            $condaOutput = Get-Content "$env:TEMP\conda_out.txt" -Raw
        }
        if (Test-Path "$env:TEMP\conda_err.txt") {
            $condaError = Get-Content "$env:TEMP\conda_err.txt" -Raw
        }
        
        Write-Info "Conda create exit code: $condaExitCode"
        
        if ($condaExitCode -ne 0) {
            Write-ErrorMessage "Conda environment creation failed with exit code: $condaExitCode"
            Write-ErrorMessage "Conda stdout:"
            Write-Host $condaOutput -ForegroundColor Yellow
            Write-ErrorMessage "Conda stderr:"
            Write-Host $condaError -ForegroundColor Red
            
            Write-ErrorMessage "Conda environment creation failed even with conda-forge channel"
            Write-Info "Please try manually running:"
            Write-Info "  conda create -y -n kwaainet python=3.10 -c conda-forge --override-channels"
            exit 1
        }
        
        # Verify environment was created
        Start-Sleep -Seconds 3  # Allow conda to finalize
        $envCheck = & $condaExe env list 2>$null
        if ($LASTEXITCODE -eq 0 -and $envCheck -match "kwaainet") {
            Write-Success "Created Python 3.10 environment for KwaaiNet"
        }
        else {
            Write-ErrorMessage "Failed to create conda environment - environment not found after creation"
            Write-ErrorMessage "Environment check output:"
            Write-Host $envCheck -ForegroundColor Red
            exit 1
        }
        
        # Clean up temp files
        Remove-Item "$env:TEMP\conda_out.txt" -ErrorAction SilentlyContinue
        Remove-Item "$env:TEMP\conda_err.txt" -ErrorAction SilentlyContinue
    }
    catch {
        Write-ErrorMessage "Failed to create conda environment: $($_.Exception.Message)"
        Write-ErrorMessage "Exception type: $($_.Exception.GetType().FullName)"
        Write-ErrorMessage "Exception details: $($_.Exception)"
        exit 1
    }
    
    Write-Success "Conda environment setup complete"
}

# Function to setup system Python with virtual environment
function Setup-SystemPython {
    Write-Step "Setting up KwaaiNet virtual environment..."
    
    try {
        $venvPath = "$script:InstallPath\venv"
        
        if (Test-Path $venvPath) {
            Write-Success "Using existing virtual environment"
        }
        else {
            Write-Info "Creating virtual environment..."
            New-Item -ItemType Directory -Path $script:InstallPath -Force | Out-Null
            & python -m venv $venvPath
            
            if ($LASTEXITCODE -ne 0) {
                Write-ErrorMessage "Failed to create virtual environment"
                exit 1
            }
            Write-Success "Created virtual environment for KwaaiNet"
        }
    }
    catch {
        Write-ErrorMessage "Failed to setup virtual environment: $($_.Exception.Message)"
        exit 1
    }
}

# Function to install system dependencies
function Install-Dependencies {
    Write-Step "Installing system dependencies..."
    
    $dependencies = @(
        @{ PackageId = "Git.Git"; DisplayName = "Git for Windows"; Required = $true },
        @{ PackageId = "Microsoft.VCRedist.2015+.x64"; DisplayName = "Visual C++ Redistributable"; Required = $true },
        @{ PackageId = "Microsoft.VisualStudio.2022.BuildTools"; DisplayName = "Visual Studio Build Tools 2022"; Required = $false }
    )
    
    foreach ($dep in $dependencies) {
        Write-Info "Checking $($dep.DisplayName)..."
        
        # Check if already installed based on package type
        $installed = $false
        
        switch ($dep.PackageId) {
            "Git.Git" {
                $installed = Test-Command "git"
            }
            "Microsoft.VCRedist.2015+.x64" {
                # Check if VC++ runtime is installed
                $installed = Test-Path "$env:SystemRoot\System32\vcruntime140.dll"
            }
            "Microsoft.VisualStudio.2022.BuildTools" {
                # Check if VS Build Tools are installed
                $installed = (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools") -or 
                            (Test-Path "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools") -or
                            (Test-Command "cl")
            }
        }
        
        if ($installed) {
            Write-Success "$($dep.DisplayName) is already installed"
        }
        else {
            $result = Install-WingetPackage -PackageId $dep.PackageId -DisplayName $dep.DisplayName
            if (-not $result -and $dep.Required) {
                Write-Warning "Failed to install required dependency: $($dep.DisplayName)"
                Write-Info "Please install $($dep.DisplayName) manually and re-run this installer"
            }
        }
    }
    
    # Refresh PATH after installations
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    
    # Check if build tools are available for package compilation
    if (-not (Test-Command "cl") -and -not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools")) {
        Write-Warning "Visual Studio Build Tools not detected."
        Write-Info "Some Python packages (like tokenizers) may fail to install if they need compilation."
        Write-Info "The installer will try to use pre-built wheels, but if installation fails, consider installing:"
        Write-Info "   - Visual Studio Build Tools 2022 with C++ support"
        Write-Info "   - Or use 'winget install Microsoft.VisualStudio.2022.BuildTools'"
    }
}

# Function to install tokenizers with fallback handling  
function Install-TokenizersWithFallback {
    param(
        [string]$PipCommand
    )
    
    Write-Step "Installing tokenizers with build fallback handling..."
    
    # Strategy 1: Try pre-built wheels first (most likely to work on Windows)
    Write-Info "Attempting to install tokenizers (pre-built wheels only)..."
    
    # Execute pip command properly
    if ($PipCommand -like "*conda run*") {
        # Handle conda run command
        $result = & conda run -n kwaainet pip install --only-binary=tokenizers "tokenizers>=0.19.0,<0.20.0" 2>$null
    } else {
        # Handle direct pip command
        $result = & $PipCommand install --only-binary=tokenizers "tokenizers>=0.19.0,<0.20.0" 2>$null
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "tokenizers installed successfully (pre-built wheels)"
        return $true
    }
    Write-Warning "Failed to install tokenizers pre-built wheels."
    
    # Strategy 2: Try latest version with pre-built wheels
    Write-Info "Attempting to install latest tokenizers (pre-built wheels only)..."
    if ($PipCommand -like "*conda run*") {
        $result = & conda run -n kwaainet pip install --only-binary=tokenizers tokenizers 2>$null
    } else {
        $result = & $PipCommand install --only-binary=tokenizers tokenizers 2>$null
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Success "tokenizers installed successfully (pre-built wheels)"
        return $true
    }
    Write-Warning "Failed to install latest tokenizers pre-built wheels."
    
    # Strategy 3: Try older stable version
    Write-Info "Attempting to install tokenizers 0.19.1 (pre-built wheels only)..."
    if ($PipCommand -like "*conda run*") {
        $result = & conda run -n kwaainet pip install --only-binary=tokenizers "tokenizers==0.19.1" 2>$null
    } else {
        $result = & $PipCommand install --only-binary=tokenizers "tokenizers==0.19.1" 2>$null
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Success "tokenizers 0.19.1 installed successfully (pre-built wheels)"
        return $true
    }
    Write-Warning "Failed to install tokenizers 0.19.1 pre-built wheels."
    
    # Strategy 4: Last resort - try conda if available
    if ($script:PythonMethod -eq "conda") {
        Write-Info "Attempting to install tokenizers via conda..."
        $result = & conda install -y tokenizers -c conda-forge 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "tokenizers installed via conda"
            return $true
        }
        Write-Warning "Failed to install tokenizers via conda."
    }
    
    Write-ErrorMessage "All tokenizers installation strategies failed."
    Write-Info "This is likely due to missing Rust compiler or build tools on Windows."
    Write-Info "Manual installation options:"
    Write-Info "   1. Install Visual Studio Build Tools with C++ support"
    Write-Info "   2. Install Rust compiler: https://rustup.rs/"
    Write-Info "   3. Force pre-built wheels: pip install --only-binary=tokenizers tokenizers"
    Write-Warning "Installation will continue, but tokenizers may not work properly."
    
    return $false
}

# Function to install Python packages
function Install-PythonPackages {
    Write-Step "Installing KwaaiNet Python packages..."
    
    try {
        if ($script:PythonMethod -eq "conda") {
            # Activate conda environment and install packages
            Write-Info "Activating conda environment and installing packages..."
            
            # Clear pip cache to ensure fresh installation
            Write-Info "Clearing pip cache for fresh installation..."
            & conda run -n kwaainet pip cache purge 2>$null
            
            # Install basic dependencies
            Write-Info "Installing basic dependencies..."
            & conda run -n kwaainet pip install pyyaml 2>$null
            
            # Install tokenizers first with fallback handling
            Install-TokenizersWithFallback -PipCommand "conda run -n kwaainet pip"
            
            # Install updated petals with rope_scaling support
            Write-Info "Installing Petals 2.3.0.dev2 with rope_scaling support..."
            Write-Info "This may take several minutes as it builds from source..."
            
            $petalsResult = & conda run -n kwaainet pip install "git+https://github.com/bigscience-workshop/petals.git" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Petals installed successfully from git"
            }
            else {
                Write-Warning "Failed to install petals from git. Trying fallback installation..."
                & conda run -n kwaainet pip install petals 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Petals installed from PyPI"
                }
                else {
                    Write-Warning "Failed to install petals. Continuing with PyTorch installation..."
                }
            }
            
            # Install compatible versions of transformers and huggingface_hub with tokenizers pinning
            Write-Info "Installing compatible transformers and huggingface_hub versions..."
            $transformersResult = & conda run -n kwaainet pip install "transformers==4.43.1" "tokenizers>=0.19.0,<0.20.0" "huggingface_hub>=0.20.0" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Successfully installed compatible transformers and huggingface_hub with tokenizers"
            }
            else {
                Write-Warning "Failed to install transformers/huggingface_hub with tokenizers pinning. Trying without tokenizers pinning..."
                & conda run -n kwaainet pip install "transformers==4.43.1" "huggingface_hub>=0.20.0" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Successfully installed transformers and huggingface_hub (without tokenizers pinning)"
                }
                else {
                    Write-Warning "Failed to install transformers/huggingface_hub. May have compatibility issues..."
                }
            }
            
            # Install PyTorch
            Write-Info "Installing PyTorch (CPU version)..."
            Write-Info "This may take a few minutes to download..."
            & conda run -n kwaainet pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PyTorch installed successfully"
            }
            else {
                Write-ErrorMessage "Failed to install PyTorch. Please check your internet connection."
                exit 1
            }
            
            # Install bitsandbytes for quantization support
            Write-Info "Installing bitsandbytes for quantization support..."
            if ($script:GpuType -eq "nvidia") {
                Write-Info "Installing CUDA-compatible version for NVIDIA GPU..."
            } else {
                Write-Info "Installing CPU version..."
            }
            & conda run -n kwaainet pip install bitsandbytes 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "bitsandbytes installed successfully"
            }
            else {
                Write-Warning "Failed to install bitsandbytes. Quantization may not work properly."
            }
            
            # Install KwaaiNet from GitHub
            Write-Info "Installing KwaaiNet from GitHub..."
            $githubUrl = "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/windows"
            & conda run -n kwaainet pip install $githubUrl 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "KwaaiNet Windows package installed successfully from GitHub"
            }
            else {
                Write-Warning "Failed to install KwaaiNet from GitHub, trying fallback installation..."
                # Try installing the main package without subdirectory
                & conda run -n kwaainet pip install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "KwaaiNet installed successfully from GitHub (main package)"
                }
                else {
                    Write-ErrorMessage "Failed to install KwaaiNet from GitHub"
                    Write-Info "This may be due to missing build tools, git, or network connectivity issues."
                    Write-Info "Please ensure git is installed and try running manually:"
                    Write-Info "  conda run -n kwaainet pip install 'git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git'"
                    exit 1
                }
            }
        }
        else {
            # Use virtual environment
            $venvPath = "$script:InstallPath\venv"
            $pipExec = "$venvPath\Scripts\pip.exe"
            $pythonExec = "$venvPath\Scripts\python.exe"
            
            Write-Info "Installing packages in virtual environment..."
            
            # Clear pip cache to ensure fresh installation  
            Write-Info "Clearing pip cache for fresh installation..."
            & $pipExec cache purge 2>$null
            
            # Install basic dependencies
            & $pipExec install pyyaml 2>$null
            
            # Install tokenizers first with fallback handling
            Install-TokenizersWithFallback -PipCommand $pipExec
            
            # Install updated petals
            Write-Info "Installing Petals 2.3.0.dev2 with rope_scaling support..."
            $petalsResult = & $pipExec install "git+https://github.com/bigscience-workshop/petals.git" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Petals installed successfully from git"
            }
            else {
                Write-Warning "Failed to install petals from git. Trying fallback..."
                & $pipExec install petals 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Failed to install petals. Continuing with PyTorch..."
                }
            }
            
            # Install compatible versions of transformers and huggingface_hub with tokenizers pinning
            Write-Info "Installing compatible transformers and huggingface_hub versions..."
            $transformersResult = & $pipExec install "transformers==4.43.1" "tokenizers>=0.19.0,<0.20.0" "huggingface_hub>=0.20.0" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install with tokenizers pinning. Trying without tokenizers pinning..."
                & $pipExec install "transformers==4.43.1" "huggingface_hub>=0.20.0" 2>$null
            }
            
            # Install PyTorch
            Write-Info "Installing PyTorch (CPU version)..."
            & $pipExec install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-ErrorMessage "Failed to install PyTorch. Please check your internet connection."
                exit 1
            }
            Write-Success "PyTorch installed successfully"
            
            # Install bitsandbytes for quantization support
            Write-Info "Installing bitsandbytes for quantization support..."
            if ($script:GpuType -eq "nvidia") {
                Write-Info "Installing CUDA-compatible version for NVIDIA GPU..."
            } else {
                Write-Info "Installing CPU version..."
            }
            & $pipExec install bitsandbytes 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "bitsandbytes installed successfully"
            }
            else {
                Write-Warning "Failed to install bitsandbytes. Quantization may not work properly."
            }
            
            # Install KwaaiNet from GitHub
            Write-Info "Installing KwaaiNet from GitHub..."
            $githubUrl = "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/windows"
            & $pipExec install $githubUrl 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "KwaaiNet Windows package installed successfully from GitHub"
            }
            else {
                Write-Warning "Failed to install KwaaiNet from GitHub, trying fallback installation..."
                # Try installing the main package without subdirectory
                & $pipExec install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "KwaaiNet installed successfully from GitHub (main package)"
                }
                else {
                    Write-ErrorMessage "Failed to install KwaaiNet from GitHub"
                    Write-Info "This may be due to missing build tools, git, or network connectivity issues."
                    Write-Info "Please ensure git is installed and try running manually:"
                    Write-Info "  pip install 'git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git'"
                    exit 1
                }
            }
        }
        
        Write-Success "Python packages installed successfully"
    }
    catch {
        Write-ErrorMessage "Failed to install Python packages: $($_.Exception.Message)"
        exit 1
    }
}

# Function to create launcher scripts
function New-LauncherScripts {
    Write-Step "Creating launcher scripts..."
    
    try {
        $launcherDir = "$script:InstallPath\bin"
        New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null
        
        if ($script:PythonMethod -eq "conda") {
            # Create PowerShell launcher for conda
            $psLauncher = @'
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first
param([Parameter(ValueFromRemainingArguments)]$Args)

# Find conda installation (prioritize user installations over system)
$condaPath = $null
if (Test-Path "$env:USERPROFILE\Miniconda3\Scripts\conda.exe") {
    $condaPath = "$env:USERPROFILE\Miniconda3"
} elseif (Test-Path "$env:USERPROFILE\Anaconda3\Scripts\conda.exe") {
    $condaPath = "$env:USERPROFILE\Anaconda3"
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    # Try to get conda base, but verify it exists
    try {
        $potentialPath = & conda info --base 2>$null
        if ($potentialPath -and (Test-Path "$potentialPath\Scripts\conda.exe")) {
            $condaPath = $potentialPath
        } else {
            # Fallback to directory detection
            $condaPath = Split-Path (Split-Path (Get-Command conda).Source)
        }
    } catch {
        $condaPath = Split-Path (Split-Path (Get-Command conda).Source)
    }
}

if (-not $condaPath -or -not (Test-Path "$condaPath\Scripts\conda.exe")) {
    Write-Error @"
[ERROR] Could not find conda installation with conda.exe.
Expected locations:
  - $env:USERPROFILE\Miniconda3\Scripts\conda.exe
  - $env:USERPROFILE\Anaconda3\Scripts\conda.exe
"@
    exit 1
}

# Activate the environment and run the command
$condaExe = "$condaPath\Scripts\conda.exe"
& $condaExe run -n kwaainet python -m kwaainet.runner @Args
'@
            
            $psLauncherPath = "$launcherDir\kwaainet.ps1"
            Set-Content -Path $psLauncherPath -Value $psLauncher -Encoding UTF8
            
            # Create batch file launcher that calls PowerShell
            $batchLauncher = @"
@echo off
powershell.exe -ExecutionPolicy Bypass -File "$psLauncherPath" %*
"@
            $batchLauncherPath = "$launcherDir\kwaainet.bat"
            Set-Content -Path $batchLauncherPath -Value $batchLauncher -Encoding ASCII
        }
        else {
            # Create batch launcher for virtual environment
            $venvPath = "$script:InstallPath\venv"
            $batchLauncher = @"
@echo off
REM KwaaiNet Launcher - Run KwaaiNet from virtual environment

set VENV_PATH=$venvPath

if not exist "%VENV_PATH%" (
    echo [ERROR] KwaaiNet virtual environment not found at %VENV_PATH%
    exit /b 1
)

REM Activate virtual environment and run the command
call "%VENV_PATH%\Scripts\activate.bat" && python -m kwaainet.runner %*
"@
            $batchLauncherPath = "$launcherDir\kwaainet.bat"
            Set-Content -Path $batchLauncherPath -Value $batchLauncher -Encoding ASCII
            
            # Also create PowerShell version
            $psLauncher = @"
# KwaaiNet Launcher - Run KwaaiNet from virtual environment
param([Parameter(ValueFromRemainingArguments)]`$Args)

`$venvPath = "$venvPath"

if (-not (Test-Path `$venvPath)) {
    Write-Error "[ERROR] KwaaiNet virtual environment not found at `$venvPath"
    exit 1
}

# Activate virtual environment and run the command
& "`$venvPath\Scripts\python.exe" -m kwaainet.runner @Args
"@
            $psLauncherPath = "$launcherDir\kwaainet.ps1"
            Set-Content -Path $psLauncherPath -Value $psLauncher -Encoding UTF8
        }
        
        Write-Success "Launcher scripts created successfully"
        Write-Info "Launchers created at:"
        Write-Host "   PowerShell: $launcherDir\kwaainet.ps1" -ForegroundColor White
        Write-Host "   Batch: $launcherDir\kwaainet.bat" -ForegroundColor White
        
        # Add to PATH
        Add-ToPath $launcherDir
        
    }
    catch {
        Write-ErrorMessage "Failed to create launcher scripts: $($_.Exception.Message)"
        exit 1
    }
}

# Function to add directory to PATH
function Add-ToPath {
    param([string]$Directory)
    
    try {
        Write-Step "Adding launcher directory to PATH..."
        
        # Get current user PATH
        $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Check if directory is already in PATH
        if ($currentPath -split ";" -contains $Directory) {
            Write-Success "Directory already in PATH"
            return
        }
        
        # Add to PATH
        $newPath = if ($currentPath) { "$currentPath;$Directory" } else { $Directory }
        [System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        
        # Update current session PATH
        $env:PATH += ";$Directory"
        
        Write-Success "Added $Directory to user PATH"
        Write-Info "Restart your terminal or PowerShell session for PATH changes to take effect"
    }
    catch {
        Write-Warning "Failed to add directory to PATH: $($_.Exception.Message)"
        Write-Info "You can manually add this directory to your PATH: $Directory"
    }
}

# Function to run initial setup
function Start-InitialSetup {
    Write-Step "Running initial setup..."
    
    try {
        $launcherPath = "$script:InstallPath\bin\kwaainet.ps1"
        
        if (Test-Path $launcherPath) {
            & powershell.exe -ExecutionPolicy Bypass -File $launcherPath setup 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Initial setup completed"
            }
            else {
                Write-Warning "Initial setup failed. You may need to run 'kwaainet setup' manually."
            }
        }
        else {
            Write-Warning "Launcher not found. Please run setup manually after installation."
        }
    }
    catch {
        Write-Warning "Initial setup failed: $($_.Exception.Message)"
        Write-Info "You may need to run 'kwaainet setup' manually."
    }
}

# Main installation function
function Start-Installation {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "KwaaiNet for Windows - One-Step Installer v$script:InstallerVersion" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "This installer will set up KwaaiNet for sharing compute on Windows" -ForegroundColor White
    Write-Host "It includes Python setup, dependencies, and environment configuration" -ForegroundColor White
    Write-Host ""
    
    # Check if running as administrator
    if (Test-Administrator) {
        Write-Warning "Running as Administrator. This installer should be run as a regular user."
        Write-Info "Some operations will use elevated privileges when needed."
        $script:UseElevated = $true
    }
    
    try {
        # Step 1: System detection
        $systemInfo = Get-SystemInfo
        
        # Step 2: Check execution policy
        Test-ExecutionPolicy
        
        # Step 3: GPU detection  
        Get-GpuInfo
        
        # Step 4: Install system dependencies
        Install-Dependencies
        
        # Step 5: Python environment setup
        Set-PythonEnvironment
        
        # Step 6: Install Python packages
        Install-PythonPackages
        
        # Step 7: Create launcher scripts
        New-LauncherScripts
        
        # Step 8: Run initial setup
        Start-InitialSetup
        
        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "[SUCCESS] KwaaiNet for Windows installation completed!" -ForegroundColor Green
        Write-Host ""
        Write-Host "[INFO] Configuration detected:" -ForegroundColor White
        Write-Host "   - OS: $($systemInfo.OSName)" -ForegroundColor White
        Write-Host "   - Architecture: $($systemInfo.Architecture)" -ForegroundColor White
        Write-Host "   - GPU: $script:GpuType $(if($script:GpuInfo) { "($script:GpuInfo)" })" -ForegroundColor White
        Write-Host "   - Python method: $script:PythonMethod" -ForegroundColor White
        Write-Host ""
        Write-Host "[NEXT] Next steps:" -ForegroundColor White
        Write-Host "   1. Complete the Python package installation" -ForegroundColor White
        Write-Host "   2. Set up launcher scripts" -ForegroundColor White
        Write-Host "   3. Configure GPU acceleration (if available)" -ForegroundColor White
        Write-Host ""
        Write-Host "[INFO] For more information, visit: https://github.com/Kwaai-AI-Lab/OpenAI-Petal" -ForegroundColor White
        Write-Host "==========================================================" -ForegroundColor Green
    }
    catch {
        Write-ErrorMessage "Installation failed: $($_.Exception.Message)"
        Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
        exit 1
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne ".") {
    Start-Installation
}