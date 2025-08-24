@echo off
REM KwaaiNet for Windows - Installer Launcher
REM This batch file launches the PowerShell installer with proper execution policy

echo ================================================
echo KwaaiNet for Windows - Installation Launcher
echo ================================================
echo.

REM Check if PowerShell 5.1+ is available
powershell.exe -Command "if ($PSVersionTable.PSVersion.Major -lt 5) { exit 1 }" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell 5.1 or newer is required.
    echo Please update PowerShell and try again.
    pause
    exit /b 1
)

REM Get the directory of this batch file
set "INSTALLER_DIR=%~dp0"
set "PS_SCRIPT=%INSTALLER_DIR%windowsinstaller.ps1"

REM Check if the PowerShell script exists
if not exist "%PS_SCRIPT%" (
    echo ERROR: windowsinstaller.ps1 not found in %INSTALLER_DIR%
    echo Please ensure both files are in the same directory.
    pause
    exit /b 1
)

echo Starting PowerShell installer...
echo.

REM Launch PowerShell installer with bypass execution policy
powershell.exe -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*

REM Check the exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation completed successfully!
) else (
    echo.
    echo Installation failed with error code %ERRORLEVEL%
)

echo.
echo Press any key to close this window...
pause >nul