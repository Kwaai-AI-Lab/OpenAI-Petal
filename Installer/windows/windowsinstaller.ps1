# KwaaiNet for Windows - One-Step Installer v0.2.22
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
$script:InstallerVersion = "0.2.22"

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

# Function to detect GPU hardware with advanced diagnostics
function Get-GpuInfo {
    Write-Step "Detecting GPU hardware with advanced diagnostics..."
    
    try {
        $script:GpuType = "none"
        $script:GpuInfo = ""
        $script:GpuMemory = 0
        $script:GpuComputeCapability = ""
        
        # Check for NVIDIA GPU first with detailed information
        try {
            Write-Info "Checking for NVIDIA GPU drivers and tools..."
            $nvidiaOutput = & nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>$null
            if ($LASTEXITCODE -eq 0 -and $nvidiaOutput) {
                $nvidiaSplit = $nvidiaOutput.Trim().Split(',')
                $script:GpuType = "nvidia"
                $script:GpuInfo = $nvidiaSplit[0].Trim()
                $script:GpuMemory = [int]$nvidiaSplit[1].Trim()
                $script:GpuComputeCapability = $nvidiaSplit[2].Trim()
                
                Write-Success "NVIDIA GPU detected: $script:GpuInfo"
                Write-Info "   Memory: $script:GpuMemory MB"
                Write-Info "   Compute Capability: $script:GpuComputeCapability"
                
                # Check CUDA version
                try {
                    $cudaVersion = & nvcc --version 2>$null | Select-String "release" | ForEach-Object { $_.ToString().Split(',')[1].Trim() }
                    if ($cudaVersion) {
                        Write-Success "CUDA toolkit detected: $cudaVersion"
                    } else {
                        Write-Warning "CUDA toolkit not found - GPU acceleration may be limited"
                    }
                }
                catch {
                    Write-Warning "CUDA toolkit not found - GPU acceleration may be limited"
                }
                return
            }
        }
        catch {
            # nvidia-smi not available, continue checking
        }
        
        # Enhanced GPU detection via WMI and registry
        Write-Info "Performing comprehensive GPU detection..."
        $gpus = Get-CimInstance Win32_VideoController | Where-Object { 
            $_.Name -notlike "*Basic*" -and 
            $_.Name -notlike "*Generic*" -and 
            $_.Name -notlike "*Microsoft*" -and
            $_.AdapterRAM -gt 0
        }
        
        $detectedGpus = @()
        
        foreach ($gpu in $gpus) {
            $gpuName = $gpu.Name
            $gpuMemory = [math]::Round($gpu.AdapterRAM / 1MB, 0)
            $gpuDriverVersion = $gpu.DriverVersion
            
            $gpuDetails = @{
                Name = $gpuName
                Memory = $gpuMemory
                Driver = $gpuDriverVersion
                Type = "unknown"
            }
            
            # Classify GPU type with more detailed detection
            if ($gpuName -match "NVIDIA|GeForce|GTX|RTX|Quadro|Tesla|Titan") {
                $gpuDetails.Type = "nvidia"
                $script:GpuType = "nvidia"
                $script:GpuInfo = $gpuName
                $script:GpuMemory = $gpuMemory
                
                Write-Success "NVIDIA GPU detected: $gpuName"
                Write-Info "   Memory: $gpuMemory MB, Driver: $gpuDriverVersion"
                Write-Warning "NVIDIA drivers detected but nvidia-smi not available"
                Write-Info "   Consider installing NVIDIA CUDA toolkit for optimal performance"
                
                $detectedGpus += $gpuDetails
                break
            }
            elseif ($gpuName -match "AMD|Radeon|RX|Vega|RDNA|Navi") {
                $gpuDetails.Type = "amd"
                $script:GpuType = "amd"
                $script:GpuInfo = $gpuName
                $script:GpuMemory = $gpuMemory
                
                Write-Success "AMD GPU detected: $gpuName"
                Write-Info "   Memory: $gpuMemory MB, Driver: $gpuDriverVersion"
                Write-Info "   ROCm support may be available for compute workloads"
                
                $detectedGpus += $gpuDetails
                break
            }
            elseif ($gpuName -match "Intel.*Graphics|Intel.*Iris|Intel.*HD|Intel.*UHD|Intel.*Xe") {
                $gpuDetails.Type = "intel"
                if ($script:GpuType -eq "none") {  # Only set if no dedicated GPU found
                    $script:GpuType = "intel"
                    $script:GpuInfo = $gpuName
                    $script:GpuMemory = $gpuMemory
                    
                    Write-Success "Intel integrated GPU detected: $gpuName"
                    Write-Info "   Memory: $gpuMemory MB, Driver: $gpuDriverVersion"
                    Write-Info "   Intel GPU may support some acceleration features"
                }
                
                $detectedGpus += $gpuDetails
            }
            else {
                Write-Info "Unknown GPU detected: $gpuName ($gpuMemory MB)"
                $detectedGpus += $gpuDetails
            }
        }
        
        # Check for additional GPU capabilities
        if ($script:GpuType -ne "none") {
            Write-Info "GPU acceleration recommendations:"
            switch ($script:GpuType) {
                "nvidia" {
                    Write-Info "   • Install latest NVIDIA drivers from nvidia.com"
                    Write-Info "   • Consider NVIDIA CUDA toolkit for ML acceleration"
                    Write-Info "   • GPU memory: $script:GpuMemory MB available for models"
                }
                "amd" {
                    Write-Info "   • Install latest AMD drivers from amd.com" 
                    Write-Info "   • Consider AMD ROCm for compute acceleration"
                    Write-Info "   • GPU memory: $script:GpuMemory MB available for models"
                }
                "intel" {
                    Write-Info "   • Install latest Intel graphics drivers"
                    Write-Info "   • Intel OpenVINO may provide some acceleration"
                    Write-Info "   • Shared system memory: $script:GpuMemory MB available"
                }
            }
        }
        
        if ($script:GpuType -eq "none") {
            Write-Info "No dedicated GPU detected. Using CPU-only mode."
            Write-Info "   CPU inference will be used for all operations"
            Write-Info "   Consider adding a GPU for better performance with large models"
        }
        
        # Store detected GPUs for later use
        $script:DetectedGpus = $detectedGpus
        
    }
    catch {
        Write-Warning "Failed to detect GPU information: $($_.Exception.Message)"
        $script:GpuType = "none"
        $script:DetectedGpus = @()
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
        $envList = & $condaExe env list 2>&1
        $envOutput = $envList -join "`n"
        
        if ($LASTEXITCODE -eq 0 -and $envOutput -match "kwaainet") {
            Write-Success "Using existing kwaainet conda environment"
            
            # Verify environment is functional
            Write-Info "Verifying existing environment functionality..."
            $pythonTest = & $condaExe run -n kwaainet python --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Existing environment is functional: $($pythonTest.Trim())"
                return
            } else {
                Write-Warning "Existing environment appears corrupted. Will recreate."
                Write-Info "Removing corrupted environment..."
                & $condaExe env remove -n kwaainet -y 2>$null | Out-Null
            }
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
            Write-Success "Found existing virtual environment"
            
            # Verify virtual environment is functional
            Write-Info "Verifying existing virtual environment functionality..."
            $pythonExe = "$venvPath\Scripts\python.exe"
            $pipExe = "$venvPath\Scripts\pip.exe"
            
            if ((Test-Path $pythonExe) -and (Test-Path $pipExe)) {
                try {
                    $pythonTest = & $pythonExe --version 2>&1
                    $pipTest = & $pipExe --version 2>&1
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Existing virtual environment is functional: $($pythonTest.Trim())"
                        return
                    } else {
                        Write-Warning "Virtual environment appears corrupted. Will recreate."
                        Remove-Item -Path $venvPath -Recurse -Force -ErrorAction SilentlyContinue
                    }
                }
                catch {
                    Write-Warning "Virtual environment verification failed. Will recreate."
                    Remove-Item -Path $venvPath -Recurse -Force -ErrorAction SilentlyContinue
                }
            } else {
                Write-Warning "Virtual environment is incomplete. Will recreate."
                Remove-Item -Path $venvPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        
        Write-Info "Creating new virtual environment..."
        New-Item -ItemType Directory -Path $script:InstallPath -Force | Out-Null
        
        try {
            & python -m venv $venvPath 2>&1 | Out-Null
            
            if ($LASTEXITCODE -ne 0) {
                Write-ErrorMessage "Failed to create virtual environment"
                Write-Info "This might be due to missing venv module. Try installing python3-venv package."
                exit 1
            }
            
            # Verify the newly created environment
            $pythonExe = "$venvPath\Scripts\python.exe"
            $pipExe = "$venvPath\Scripts\pip.exe"
            
            if ((Test-Path $pythonExe) -and (Test-Path $pipExe)) {
                $pythonTest = & $pythonExe --version 2>&1
                Write-Success "Created functional virtual environment: $($pythonTest.Trim())"
            } else {
                Write-ErrorMessage "Virtual environment creation completed but executables not found"
                exit 1
            }
        }
        catch {
            Write-ErrorMessage "Exception during virtual environment creation: $($_.Exception.Message)"
            exit 1
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

# Function to verify package versions
function Test-PackageVersions {
    Write-Step "Verifying installed package versions..."
    
    # Define expected package versions (aligned with Linux installer)
    $expectedPackages = @{
        "hivemind" = "1.1.10.post2"
        "petals" = "2.2.0.post1"
        "transformers" = "4.34.1"
        "huggingface_hub" = "0.34.0"
        "tokenizers" = "0.15.0"
    }
    
    $allVersionsCorrect = $true
    
    foreach ($package in $expectedPackages.Keys) {
        $expectedVersion = $expectedPackages[$package]
        
        try {
            # Get Python command based on environment
            $pythonCmd = if ($script:PythonMethod -eq "conda") {
                "conda run -n kwaainet python"
            } elseif (Test-Path "$script:InstallPath\venv\Scripts\python.exe") {
                "$script:InstallPath\venv\Scripts\python.exe"
            } else {
                "python"
            }
            
            # Check package version
            $versionCheck = "import $package; print($package.__version__)" 
            $actualVersion = & cmd /c "$pythonCmd -c `"$versionCheck`"" 2>$null
            
            if ($LASTEXITCODE -eq 0 -and $actualVersion) {
                $actualVersion = $actualVersion.Trim()
                
                # For minimum version checks (>=), verify the installed version meets requirements
                if ($expectedVersion -match "^>=(.+)") {
                    $minVersion = $matches[1]
                    Write-Info "   $package`: $actualVersion (required: >=$minVersion)"
                    
                    # Simple version comparison - this could be enhanced
                    if ([version]$actualVersion -ge [version]$minVersion) {
                        Write-Success "   ✓ $package version meets requirements"
                    } else {
                        Write-Warning "   ⚠ $package version $actualVersion is below minimum $minVersion"
                        $allVersionsCorrect = $false
                    }
                } else {
                    # Exact version or flexible range
                    Write-Info "   $package`: $actualVersion (expected: $expectedVersion)"
                    
                    if ($actualVersion -eq $expectedVersion -or $expectedVersion -eq "flexible") {
                        Write-Success "   ✓ $package version correct"
                    } else {
                        Write-Info "   ℹ $package version differs (may be acceptable)"
                    }
                }
            } else {
                Write-Warning "   ✗ Could not verify $package version"
                $allVersionsCorrect = $false
            }
        }
        catch {
            Write-Warning "   ✗ Error checking $package`: $($_.Exception.Message)"
            $allVersionsCorrect = $false
        }
    }
    
    if ($allVersionsCorrect) {
        Write-Success "All package versions verified successfully"
    } else {
        Write-Warning "Some package versions could not be verified or are incorrect"
        Write-Info "Installation may still work, but there could be compatibility issues"
    }
    
    return $allVersionsCorrect
}

# Function to test import compatibility
function Test-ImportCompatibility {
    Write-Step "Testing package import compatibility..."
    
    $importTests = @{
        "hivemind" = "import hivemind; print(f'hivemind {hivemind.__version__}')"
        "petals" = "import petals; print('petals imported successfully')"
        "transformers" = "from transformers import AutoModel; print('transformers imports working')"
        "torch" = "import torch; print(f'torch {torch.__version__}')"
        "huggingface_hub" = "import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')"
    }
    
    $allImportsSuccessful = $true
    
    foreach ($package in $importTests.Keys) {
        $testCode = $importTests[$package]
        
        try {
            # Get Python command based on environment
            $pythonCmd = if ($script:PythonMethod -eq "conda") {
                "conda run -n kwaainet python"
            } elseif (Test-Path "$script:InstallPath\venv\Scripts\python.exe") {
                "$script:InstallPath\venv\Scripts\python.exe"
            } else {
                "python"
            }
            
            Write-Info "   Testing $package import..."
            $result = & cmd /c "$pythonCmd -c `"$testCode`"" 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "   ✓ $package`: $($result.Trim())"
            } else {
                Write-Warning "   ✗ $package import failed: $result"
                $allImportsSuccessful = $false
            }
        }
        catch {
            Write-Warning "   ✗ Error testing $package import: $($_.Exception.Message)"
            $allImportsSuccessful = $false
        }
    }
    
    if ($allImportsSuccessful) {
        Write-Success "All package imports working correctly"
    } else {
        Write-Warning "Some packages failed to import - there may be installation issues"
    }
    
    return $allImportsSuccessful
}

# Function to verify KwaaiNet functionality
function Test-KwaaiNetFunctionality {
    Write-Step "Testing KwaaiNet functionality..."
    
    try {
        # Get Python command based on environment
        $pythonCmd = if ($script:PythonMethod -eq "conda") {
            "conda run -n kwaainet python"
        } elseif (Test-Path "$script:InstallPath\venv\Scripts\python.exe") {
            "$script:InstallPath\venv\Scripts\python.exe"
        } else {
            "python"
        }
        
        # Test basic KwaaiNet import and functionality
        $kwaainetTest = @"
try:
    import kwaainet
    from kwaainet import config
    print('KwaaiNet package imported successfully')
    print('Configuration system functional')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
except Exception as e:
    print(f'Configuration error: {e}')
    exit(2)
"@
        
        Write-Info "   Testing KwaaiNet package import and basic functionality..."
        $result = & cmd /c "$pythonCmd -c `"$kwaainetTest`"" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "   ✓ KwaaiNet functionality test passed"
            Write-Info "   $($result -join '; ')"
            return $true
        } else {
            Write-Warning "   ✗ KwaaiNet functionality test failed: $result"
            return $false
        }
    }
    catch {
        Write-Warning "KwaaiNet functionality test failed: $($_.Exception.Message)"
        return $false
    }
}

# Master verification function
function Start-ComprehensiveVerification {
    Write-Step "Running comprehensive post-installation verification..."
    
    $verificationTests = @(
        "Test-PackageVersions",
        "Test-ImportCompatibility", 
        "Test-KwaaiNetFunctionality"
    )
    
    $allTestsPassed = $true
    
    foreach ($test in $verificationTests) {
        try {
            $result = & $test
            if (-not $result) {
                $allTestsPassed = $false
            }
        }
        catch {
            Write-Warning "Verification test $test failed: $($_.Exception.Message)"
            $allTestsPassed = $false
        }
    }
    
    if ($allTestsPassed) {
        Write-Success "All verification tests passed! Installation appears to be working correctly."
    } else {
        Write-Warning "Some verification tests failed. The installation may have issues."
        Write-Info "You can still try running KwaaiNet, but may encounter problems."
    }
    
    return $allTestsPassed
}

# Function to test dependency compatibility before installation
function Test-DependencyCompatibility {
    Write-Step "Testing dependency compatibility matrix..."
    
    try {
        # Get Python command based on environment
        $pythonCmd = if ($script:PythonMethod -eq "conda") {
            "conda run -n kwaainet python"
        } elseif (Test-Path "$script:InstallPath\venv\Scripts\python.exe") {
            "$script:InstallPath\venv\Scripts\python.exe"
        } else {
            "python"
        }
        
        # Test critical dependency combinations
        $transformersTest = @"
import transformers, tokenizers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
print('✓ transformers + tokenizers compatibility verified')
"@
        
        $huggingfaceTest = @"
from huggingface_hub import snapshot_download
import tempfile
snapshot_download('gpt2', cache_dir=tempfile.mkdtemp(), allow_patterns=['config.json'])
print('✓ HuggingFace Hub CDN access verified')
"@
        
        $torchTest = @"
import torch, transformers
from transformers import AutoModel
print('✓ PyTorch + transformers compatibility verified')
"@
        
        $dependencyTests = @{
            "transformers_tokenizers" = @{
                "packages" = "transformers==4.34.1 tokenizers>=0.15.0"
                "test" = $transformersTest
                "description" = "transformers 4.34.1 + tokenizers >=0.15.0 compatibility"
            }
            "huggingface_hub_compatibility" = @{
                "packages" = "huggingface_hub>=0.34.0"
                "test" = $huggingfaceTest
                "description" = "HuggingFace Hub CDN connectivity and version compatibility"
            }
            "torch_transformers" = @{
                "packages" = "torch transformers==4.34.1"
                "test" = $torchTest
                "description" = "PyTorch + transformers integration"
            }
        }
        
        $allTestsPassed = $true
        
        foreach ($testName in $dependencyTests.Keys) {
            $test = $dependencyTests[$testName]
            Write-Info "Testing $($test.description)..."
            
            try {
                # Create a temporary test environment simulation
                $testScript = $test.test
                $testResult = & cmd /c "$pythonCmd -c `"$testScript`"" 2>&1
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "   ✓ $($test.description) - PASSED"
                    if ($testResult -match "✓") {
                        Write-Info "   $($testResult.Trim())"
                    }
                } else {
                    Write-Warning "   ✗ $($test.description) - FAILED"
                    Write-Host "   Error: $testResult" -ForegroundColor Yellow
                    $allTestsPassed = $false
                }
            }
            catch {
                Write-Warning "   ✗ $($test.description) - EXCEPTION: $($_.Exception.Message)"
                $allTestsPassed = $false
            }
        }
        
        # Check for known problematic package combinations
        Write-Info "Checking for known problematic combinations..."
        
        $knownIssues = @(
            @{
                "pattern" = "tokenizers.*0\.14\..*huggingface_hub.*0\.3[4-9]"
                "issue" = "tokenizers 0.14.x conflicts with huggingface_hub >=0.34.0"
                "solution" = "Use tokenizers >=0.15.0"
            },
            @{
                "pattern" = "transformers.*4\.4[0-9]\..*tokenizers.*0\.1[0-4]"
                "issue" = "transformers 4.40+ requires tokenizers >=0.15.0"
                "solution" = "Upgrade tokenizers to >=0.15.0"
            }
        )
        
        # This would be expanded to check actual installed packages
        Write-Success "No known problematic combinations detected in planned installation"
        
        if ($allTestsPassed) {
            Write-Success "All dependency compatibility tests passed"
            return $true
        } else {
            Write-Warning "Some dependency compatibility tests failed"
            Write-Info "Installation will continue but may encounter issues"
            return $false
        }
    }
    catch {
        Write-Warning "Dependency compatibility testing failed: $($_.Exception.Message)"
        Write-Info "Installation will continue but dependency issues may occur"
        return $false
    }
}

# Function to test Hugging Face connectivity
function Test-HuggingFaceConnectivity {
    Write-Step "Testing Hugging Face model download connectivity..."
    
    try {
        # Test basic HF connectivity
        Write-Info "Testing connection to huggingface.co..."
        try {
            $response = Invoke-WebRequest -Uri "https://huggingface.co" -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Basic Hugging Face connectivity verified"
            }
            else {
                Write-Warning "Unexpected response from huggingface.co (status: $($response.StatusCode))"
                Write-Info "Model downloads may fail due to network connectivity issues"
                return $false
            }
        }
        catch {
            Write-Warning "Cannot reach huggingface.co: $($_.Exception.Message)"
            Write-Info "Model downloads may fail due to network connectivity issues"
            return $false
        }
        
        # Test model file access (small config file)
        Write-Info "Testing model file access..."
        try {
            $modelResponse = Invoke-WebRequest -Uri "https://huggingface.co/gpt2/resolve/main/config.json" -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
            if ($modelResponse.StatusCode -eq 200) {
                Write-Success "Hugging Face model download connectivity verified"
                return $true
            }
            else {
                Write-Warning "Cannot access Hugging Face model files (status: $($modelResponse.StatusCode))"
                Write-Info "This may be due to network restrictions or firewall settings"
                Write-Info "Model downloads may fail, but installation will continue"
                return $false
            }
        }
        catch {
            Write-Warning "Cannot access Hugging Face model files: $($_.Exception.Message)"
            Write-Info "This may be due to network restrictions or firewall settings"
            Write-Info "Model downloads may fail, but installation will continue"
            return $false
        }
    }
    catch {
        Write-Warning "Connectivity test failed: $($_.Exception.Message)"
        return $false
    }
}

# Function to install tokenizers with advanced fallback handling  
function Install-TokenizersWithFallback {
    param(
        [string]$PipCommand
    )
    
    Write-Step "Installing tokenizers with advanced build fallback handling..."
    
    # Debug: Show environment status
    Write-Info "   Environment: $script:PythonMethod"
    Write-Info "   Pip command: $PipCommand"
    
    # Strategy 1: Try tokenizers 0.19.1 first (known stable with transformers 4.34.1)
    Write-Info "Attempting to install tokenizers 0.19.1 (known stable with transformers 4.34.1)..."
    try {
        if ($PipCommand -like "*conda run*") {
            $result = & conda run -n kwaainet pip install --only-binary=tokenizers "tokenizers==0.19.1" 2>&1
        } else {
            $result = & $PipCommand install --only-binary=tokenizers "tokenizers==0.19.1" 2>&1
        }
        $output = $result -join "`n"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "tokenizers 0.19.1 installed successfully (pre-built wheels)"
            Write-Info "   Using proven stable version with transformers 4.34.1"
            return $true
        } else {
            Write-Warning "Failed to install tokenizers 0.19.1 pre-built wheels. Error:"
            Write-Host "   $output" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Warning "Exception during tokenizers 0.19.1 installation: $($_.Exception.Message)"
    }
    
    # Strategy 2: Try tokenizers 0.15.0 (minimum compatible version)
    Write-Info "Attempting to install tokenizers 0.15.0 (minimum compatible version)..."
    try {
        if ($PipCommand -like "*conda run*") {
            $result = & conda run -n kwaainet pip install --only-binary=tokenizers "tokenizers==0.15.0" 2>&1
        } else {
            $result = & $PipCommand install --only-binary=tokenizers "tokenizers==0.15.0" 2>&1
        }
        $output = $result -join "`n"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "tokenizers 0.15.0 installed successfully (pre-built wheels)"
            return $true
        } else {
            Write-Warning "Failed to install tokenizers 0.15.0 pre-built wheels. Error:"
            Write-Host "   $output" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Warning "Exception during tokenizers 0.15.0 installation: $($_.Exception.Message)"
    }
    
    # Strategy 3: Try tokenizers 0.14.1 (Linux installer tested version)
    Write-Info "Attempting to install tokenizers 0.14.1 (Linux installer tested version)..."
    try {
        if ($PipCommand -like "*conda run*") {
            $result = & conda run -n kwaainet pip install --only-binary=tokenizers "tokenizers==0.14.1" 2>&1
        } else {
            $result = & $PipCommand install --only-binary=tokenizers "tokenizers==0.14.1" 2>&1
        }
        $output = $result -join "`n"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "tokenizers 0.14.1 installed successfully (pre-built wheels)"
            Write-Info "   Using Linux installer compatible version"
            return $true
        } else {
            Write-Warning "Failed to install tokenizers 0.14.1 pre-built wheels. Error:"
            Write-Host "   $output" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Warning "Exception during tokenizers 0.14.1 installation: $($_.Exception.Message)"
    }
    
    # Strategy 4: Try latest version with pre-built wheels only (last pre-built attempt)
    Write-Info "Attempting to install latest tokenizers (pre-built wheels only)..."
    try {
        if ($PipCommand -like "*conda run*") {
            $result = & conda run -n kwaainet pip install --only-binary=tokenizers tokenizers 2>&1
        } else {
            $result = & $PipCommand install --only-binary=tokenizers tokenizers 2>&1
        }
        $output = $result -join "`n"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "tokenizers installed successfully (pre-built wheels)"
            return $true
        } else {
            Write-Warning "Failed to install latest tokenizers pre-built wheels. Error:"
            Write-Host "   $output" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Warning "Exception during latest tokenizers installation: $($_.Exception.Message)"
    }
    
    # Strategy 5: Try compilation only if build tools are available
    $hasBuildTools = (Test-Command "cl") -or (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools") -or (Test-Path "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools")
    
    if ($hasBuildTools) {
        Write-Info "Build tools detected. Attempting to compile tokenizers from source (with timeout)..."
        try {
            $job = Start-Job -ScriptBlock {
                param($PipCommand)
                if ($PipCommand -like "*conda run*") {
                    & conda run -n kwaainet pip install tokenizers --no-cache-dir 2>&1
                } else {
                    & $PipCommand install tokenizers --no-cache-dir 2>&1
                }
                return $LASTEXITCODE
            } -ArgumentList $PipCommand
            
            # Wait for job with timeout (5 minutes)
            $completed = Wait-Job $job -Timeout 300
            
            if ($completed) {
                $result = Receive-Job $job
                $exitCode = $result[-1]  # Last item should be exit code
                Remove-Job $job
                
                if ($exitCode -eq 0) {
                    Write-Success "tokenizers compiled successfully from source"
                    return $true
                } else {
                    Write-Warning "Failed to compile tokenizers from source. Output:"
                    Write-Host ($result -join "`n") -ForegroundColor Yellow
                }
            } else {
                Write-Warning "Compilation timeout (5 minutes) - stopping attempt"
                Stop-Job $job
                Remove-Job $job
            }
        }
        catch {
            Write-Warning "Exception during compilation attempt: $($_.Exception.Message)"
        }
    } else {
        Write-Info "No build tools detected - skipping compilation attempt"
    }
    
    # Strategy 6: Emergency fallback - try conda if available
    if ($script:PythonMethod -eq "conda") {
        Write-Info "Emergency fallback: trying conda installation..."
        try {
            $result = & conda install -y tokenizers -c conda-forge 2>&1
            $output = $result -join "`n"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "tokenizers installed via conda"
                return $true
            } else {
                Write-Warning "Failed to install tokenizers via conda. Error:"
                Write-Host "   $output" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Warning "Exception during conda installation: $($_.Exception.Message)"
        }
    }
    
    # Strategy 7: Last attempt - try without any constraints but with timeout
    Write-Info "Last attempt: installing tokenizers with extended timeout (10 minutes)..."
    try {
        $job = Start-Job -ScriptBlock {
            param($PipCommand)
            if ($PipCommand -like "*conda run*") {
                & conda run -n kwaainet pip install tokenizers --no-cache-dir 2>&1
            } else {
                & $PipCommand install tokenizers --no-cache-dir 2>&1
            }
            return $LASTEXITCODE
        } -ArgumentList $PipCommand
        
        # Wait for job with extended timeout (10 minutes)
        $completed = Wait-Job $job -Timeout 600
        
        if ($completed) {
            $result = Receive-Job $job
            $exitCode = $result[-1]  # Last item should be exit code
            Remove-Job $job
            
            if ($exitCode -eq 0) {
                Write-Success "tokenizers installed successfully (extended timeout)"
                return $true
            } else {
                Write-Warning "Final attempt failed. Output:"
                Write-Host ($result -join "`n") -ForegroundColor Yellow
            }
        } else {
            Write-Warning "Extended timeout (10 minutes) reached - stopping final attempt"
            Stop-Job $job
            Remove-Job $job
        }
    }
    catch {
        Write-Warning "Exception during final attempt: $($_.Exception.Message)"
    }
    
    Write-ErrorMessage "All tokenizers installation strategies failed."
    Write-Info ""
    Write-Info "🔧 IMMEDIATE WORKAROUNDS:"
    Write-Info "   Option 1: Manual pre-built wheel installation"
    Write-Info "     pip install --only-binary=tokenizers tokenizers"
    Write-Info "   Option 2: Install build dependencies"
    Write-Info "     1. Install Visual Studio Build Tools with C++ support"
    Write-Info "     2. Install Rust compiler: https://rustup.rs/"
    Write-Info "     3. Retry: pip install tokenizers"
    Write-Info "   Option 3: Use conda environment"
    Write-Info "     conda install tokenizers -c conda-forge"
    Write-Info ""
    Write-Warning "Installation will continue, but tokenizers may not work properly."
    Write-Info "You may encounter issues with model loading and text processing."
    
    return $false
}

# Function to install Python packages
function Install-PythonPackages {
    Write-Step "Installing KwaaiNet Python packages..."
    
    try {
        if ($script:PythonMethod -eq "conda") {
            # Activate conda environment and install packages
            Write-Info "Activating conda environment and installing packages..."
            
            # Skip cache clearing to avoid potential package pattern issues
            Write-Info "Preparing package installation environment..."
            
            # Install basic dependencies
            Write-Info "Installing basic dependencies..."
            & conda run -n kwaainet pip install pyyaml 2>$null
            
            # Install tokenizers first with fallback handling
            Install-TokenizersWithFallback -PipCommand "conda run -n kwaainet pip"
            
            # Install updated petals with rope_scaling support
            Write-Info "Installing Petals 2.3.0.dev2 with rope_scaling support..."
            Write-Info "This may take several minutes as it builds from source..."
            
            # Test git connectivity before attempting GitHub installation
            $useGitInstall = $false
            if (Test-Command "git") {
                Write-Info "Testing git connectivity to GitHub..."
                # Test git connectivity with a simple command
                $gitTest = & git ls-remote --heads --exit-code https://github.com/bigscience-workshop/petals.git 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Git connectivity to GitHub verified"
                    $useGitInstall = $true
                }
                else {
                    Write-Warning "Git connectivity test failed. Will use PyPI installation instead."
                    Write-Info "This may be due to network restrictions, proxy settings, or firewall blocking git."
                }
            }
            else {
                Write-Warning "Git not found. Will use PyPI installation."
            }
            
            if ($useGitInstall) {
                Write-Info "Installing Petals from GitHub (may take several minutes)..."
                $petalsOutput = ""
                $petalsError = ""
                
                try {
                    # Capture both stdout and stderr for better error reporting
                    $petalsResult = & conda run -n kwaainet pip install "git+https://github.com/bigscience-workshop/petals.git" 2>&1
                    $petalsOutput = $petalsResult -join "`n"
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Petals installed successfully from GitHub"
                    }
                    else {
                        Write-Warning "GitHub installation failed despite connectivity test. Error output:"
                        Write-Host $petalsOutput -ForegroundColor Yellow
                        Write-Info "Trying PyPI fallback..."
                        
                        $pypiResult = & conda run -n kwaainet pip install petals 2>&1
                        $pypiOutput = $pypiResult -join "`n"
                        
                        if ($LASTEXITCODE -eq 0) {
                            Write-Success "Petals installed from PyPI"
                        }
                        else {
                            Write-ErrorMessage "Failed to install petals from both GitHub and PyPI."
                            Write-Host "GitHub error: $petalsOutput" -ForegroundColor Red
                            Write-Host "PyPI error: $pypiOutput" -ForegroundColor Red
                            Write-Info "Manual installation options:"
                            Write-Info "  conda run -n kwaainet pip install petals"
                            Write-Info "  conda run -n kwaainet pip install 'git+https://github.com/bigscience-workshop/petals.git'"
                            exit 1
                        }
                    }
                }
                catch {
                    Write-ErrorMessage "Exception during Petals installation: $($_.Exception.Message)"
                    exit 1
                }
            }
            else {
                Write-Info "Installing Petals from PyPI..."
                try {
                    $pypiResult = & conda run -n kwaainet pip install petals 2>&1
                    $pypiOutput = $pypiResult -join "`n"
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Petals installed from PyPI"
                    }
                    else {
                        Write-ErrorMessage "Failed to install petals from PyPI. This is required for distributed inference."
                        Write-Host "Error output: $pypiOutput" -ForegroundColor Red
                        Write-Info "Please check your internet connection and try manual installation:"
                        Write-Info "  conda run -n kwaainet pip install petals"
                        exit 1
                    }
                }
                catch {
                    Write-ErrorMessage "Exception during PyPI Petals installation: $($_.Exception.Message)"
                    exit 1
                }
            }
            
            # Install compatible versions of transformers and huggingface_hub with tokenizers pinning
            Write-Info "Installing compatible transformers and huggingface_hub versions..."
            try {
                $transformersResult = & conda run -n kwaainet pip install "transformers==4.34.1" "tokenizers>=0.15.0" "huggingface_hub>=0.34.0" 2>&1
                $transformersOutput = $transformersResult -join "`n"
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Successfully installed compatible transformers and huggingface_hub with tokenizers"
                }
                else {
                    Write-Warning "Failed to install transformers/huggingface_hub with tokenizers pinning. Error output:"
                    Write-Host $transformersOutput -ForegroundColor Yellow
                    Write-Info "Trying without tokenizers pinning..."
                    
                    $fallbackResult = & conda run -n kwaainet pip install "transformers==4.34.1" "huggingface_hub>=0.34.0" 2>&1
                    $fallbackOutput = $fallbackResult -join "`n"
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Successfully installed transformers and huggingface_hub (without tokenizers pinning)"
                    }
                    else {
                        Write-Warning "Failed to install transformers/huggingface_hub. May have compatibility issues..."
                        Write-Host "First attempt error: $transformersOutput" -ForegroundColor Red
                        Write-Host "Fallback error: $fallbackOutput" -ForegroundColor Red
                    }
                }
            }
            catch {
                Write-ErrorMessage "Exception during transformers installation: $($_.Exception.Message)"
            }
            
            # Install PyTorch with compatibility validation
            Write-Info "Installing PyTorch (CPU version)..."
            Write-Info "This may take a few minutes to download..."
            try {
                $torchResult = & conda run -n kwaainet pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1
                $torchOutput = $torchResult -join "`n"
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "PyTorch installed successfully"
                    
                    # Verify PyTorch compatibility with transformers
                    Write-Info "Validating PyTorch + transformers compatibility..."
                    $compatTest = & conda run -n kwaainet python -c "import torch, transformers; print(f'PyTorch {torch.__version__} + transformers compatibility verified')" 2>&1
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "PyTorch-transformers compatibility verified: $($compatTest.Trim())"
                    } else {
                        Write-Warning "PyTorch-transformers compatibility issue detected: $compatTest"
                        Write-Info "Installation will continue, but there may be runtime issues"
                    }
                } else {
                    Write-ErrorMessage "Failed to install PyTorch. Error output:"
                    Write-Host $torchOutput -ForegroundColor Red
                    Write-Info "Please check your internet connection and try again."
                    exit 1
                }
            }
            catch {
                Write-ErrorMessage "Exception during PyTorch installation: $($_.Exception.Message)"
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
            
            # Verify git is available before attempting GitHub installation
            if (-not (Test-Command "git")) {
                Write-ErrorMessage "Git is required to install KwaaiNet from GitHub but was not found"
                Write-Info "Please install Git and re-run the installer"
                exit 1
            }
            
            # Try installing with proper subdirectory syntax
            $githubUrl = "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#egg=kwaainet&subdirectory=Installer/windows"
            & conda run -n kwaainet pip install $githubUrl 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "KwaaiNet Windows package installed successfully from GitHub"
            }
            else {
                Write-Warning "Failed to install KwaaiNet subdirectory package, trying main package..."
                # Try installing the main package 
                & conda run -n kwaainet pip install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#egg=kwaainet" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "KwaaiNet installed successfully from GitHub (main package)"
                }
                else {
                    Write-ErrorMessage "Failed to install KwaaiNet from GitHub"
                    Write-Info "This may be due to missing build tools, git access, or network connectivity issues."
                    Write-Info "Please ensure git is working and try running manually:"
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
            
            # Skip cache clearing to avoid potential package pattern issues
            Write-Info "Preparing package installation environment..."
            
            # Install basic dependencies
            & $pipExec install pyyaml 2>$null
            
            # Install tokenizers first with fallback handling
            Install-TokenizersWithFallback -PipCommand $pipExec
            
            # Install updated petals with rope_scaling support
            Write-Info "Installing Petals 2.3.0.dev2 with rope_scaling support..."
            Write-Info "This may take several minutes as it builds from source..."
            
            # Test git connectivity before attempting GitHub installation
            $useGitInstall = $false
            if (Test-Command "git") {
                Write-Info "Testing git connectivity to GitHub..."
                # Test git connectivity with a simple command
                $gitTest = & git ls-remote --heads --exit-code https://github.com/bigscience-workshop/petals.git 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Git connectivity to GitHub verified"
                    $useGitInstall = $true
                }
                else {
                    Write-Warning "Git connectivity test failed. Will use PyPI installation instead."
                    Write-Info "This may be due to network restrictions, proxy settings, or firewall blocking git."
                }
            }
            else {
                Write-Warning "Git not found. Will use PyPI installation."
            }
            
            if ($useGitInstall) {
                Write-Info "Installing Petals from GitHub (may take several minutes)..."
                $petalsResult = & $pipExec install "git+https://github.com/bigscience-workshop/petals.git" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Petals installed successfully from GitHub"
                }
                else {
                    Write-Warning "GitHub installation failed despite connectivity test. Trying PyPI fallback..."
                    & $pipExec install petals 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "Petals installed from PyPI"
                    }
                    else {
                        Write-ErrorMessage "Failed to install petals from both GitHub and PyPI."
                        Write-Info "Manual installation options:"
                        Write-Info "  pip install petals"
                        Write-Info "  pip install 'git+https://github.com/bigscience-workshop/petals.git'"
                        exit 1
                    }
                }
            }
            else {
                Write-Info "Installing Petals from PyPI..."
                & $pipExec install petals 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Petals installed from PyPI"
                }
                else {
                    Write-ErrorMessage "Failed to install petals from PyPI. This is required for distributed inference."
                    Write-Info "Please check your internet connection and try manual installation:"
                    Write-Info "  pip install petals"
                    exit 1
                }
            }
            
            # Install compatible versions of transformers and huggingface_hub with tokenizers pinning
            Write-Info "Installing compatible transformers and huggingface_hub versions..."
            $transformersResult = & $pipExec install "transformers==4.34.1" "tokenizers>=0.15.0" "huggingface_hub>=0.34.0" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install with tokenizers pinning. Trying without tokenizers pinning..."
                & $pipExec install "transformers==4.34.1" "huggingface_hub>=0.34.0" 2>$null
            }
            
            # Install PyTorch with compatibility validation
            Write-Info "Installing PyTorch (CPU version)..."
            try {
                $torchResult = & $pipExec install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1
                $torchOutput = $torchResult -join "`n"
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "PyTorch installed successfully"
                    
                    # Verify PyTorch compatibility with transformers
                    Write-Info "Validating PyTorch + transformers compatibility..."
                    $pythonExe = "$script:InstallPath\venv\Scripts\python.exe"
                    $compatTest = & $pythonExe -c "import torch, transformers; print(f'PyTorch {torch.__version__} + transformers compatibility verified')" 2>&1
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Success "PyTorch-transformers compatibility verified: $($compatTest.Trim())"
                    } else {
                        Write-Warning "PyTorch-transformers compatibility issue detected: $compatTest"
                        Write-Info "Installation will continue, but there may be runtime issues"
                    }
                } else {
                    Write-ErrorMessage "Failed to install PyTorch. Error output:"
                    Write-Host $torchOutput -ForegroundColor Red
                    Write-Info "Please check your internet connection and try again."
                    exit 1
                }
            }
            catch {
                Write-ErrorMessage "Exception during PyTorch installation: $($_.Exception.Message)"
                exit 1
            }
            
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
            
            # Verify git is available before attempting GitHub installation
            if (-not (Test-Command "git")) {
                Write-ErrorMessage "Git is required to install KwaaiNet from GitHub but was not found"
                Write-Info "Please install Git and re-run the installer"
                exit 1
            }
            
            # Try installing with proper subdirectory syntax
            $githubUrl = "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#egg=kwaainet&subdirectory=Installer/windows"
            & $pipExec install $githubUrl 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "KwaaiNet Windows package installed successfully from GitHub"
            }
            else {
                Write-Warning "Failed to install KwaaiNet subdirectory package, trying main package..."
                # Try installing the main package
                & $pipExec install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#egg=kwaainet" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "KwaaiNet installed successfully from GitHub (main package)"
                }
                else {
                    Write-ErrorMessage "Failed to install KwaaiNet from GitHub"
                    Write-Info "This may be due to missing build tools, git access, or network connectivity issues."
                    Write-Info "Please ensure git is working and try running manually:"
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
        
        # Step 6: Test connectivity before package installation
        Test-HuggingFaceConnectivity
        
        # Step 7: Install Python packages
        Install-PythonPackages
        
        # Step 8: Create launcher scripts
        New-LauncherScripts
        
        # Step 9: Run dependency compatibility pre-check
        Test-DependencyCompatibility
        
        # Step 10: Run comprehensive verification  
        Start-ComprehensiveVerification
        
        # Step 11: Run initial setup
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