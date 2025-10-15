"""
Service management for auto-start functionality on macOS using launchd.
"""

import os
import subprocess
import pwd
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MacOSServiceManager:
    """Manages macOS launchd service for KwaaiNet auto-start"""
    
    def __init__(self):
        self.service_name = "ai.kwaai.kwaainet"
        self.username = pwd.getpwuid(os.getuid()).pw_name
        self.home_dir = os.path.expanduser("~")
        self.launch_agents_dir = os.path.join(self.home_dir, "Library/LaunchAgents")
        self.plist_path = os.path.join(self.launch_agents_dir, f"{self.service_name}.plist")
        self.kwaainet_path = os.path.join(self.home_dir, ".local/bin/kwaainet")
        
    def _create_plist_xml(self) -> str:
        """Create the launchd plist XML content"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{self.kwaainet_path}</string>
        <string>start</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.join(self.home_dir, '.kwaainet/logs/service.log')}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.join(self.home_dir, '.kwaainet/logs/service.error.log')}</string>
    <key>WorkingDirectory</key>
    <string>{self.home_dir}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>'''
    
    def install_service(self) -> bool:
        """Install the launchd service for auto-start"""
        try:
            # Create LaunchAgents directory if it doesn't exist
            os.makedirs(self.launch_agents_dir, exist_ok=True)
            
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(self.home_dir, '.kwaainet/logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Check if kwaainet launcher exists
            if not os.path.exists(self.kwaainet_path):
                logger.error(f"KwaaiNet launcher not found at {self.kwaainet_path}")
                return False
            
            # Create plist XML content
            plist_xml = self._create_plist_xml()
            
            # Write plist file
            with open(self.plist_path, 'w') as f:
                f.write(plist_xml)
            
            # Set proper permissions
            os.chmod(self.plist_path, 0o644)
            
            # Load the service
            result = subprocess.run([
                'launchctl', 'load', self.plist_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to load service: {result.stderr}")
                return False
            
            logger.info(f"Successfully installed KwaaiNet auto-start service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return False
    
    def uninstall_service(self) -> bool:
        """Uninstall the launchd service"""
        try:
            # Unload the service if it's loaded
            if self.is_service_loaded():
                result = subprocess.run([
                    'launchctl', 'unload', self.plist_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Warning while unloading service: {result.stderr}")
            
            # Remove plist file
            if os.path.exists(self.plist_path):
                os.remove(self.plist_path)
                logger.info("Removed service plist file")
            
            logger.info("Successfully uninstalled KwaaiNet auto-start service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return False
    
    def is_service_installed(self) -> bool:
        """Check if the service plist file exists"""
        return os.path.exists(self.plist_path)
    
    def is_service_loaded(self) -> bool:
        """Check if the service is currently loaded in launchctl"""
        try:
            result = subprocess.run([
                'launchctl', 'list', self.service_name
            ], capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error checking service status: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        status = {
            'installed': self.is_service_installed(),
            'loaded': False,
            'running': False,
            'pid': None,
            'exit_code': None
        }
        
        if status['installed']:
            status['loaded'] = self.is_service_loaded()
            
            if status['loaded']:
                try:
                    # Get detailed status from launchctl
                    result = subprocess.run([
                        'launchctl', 'list', self.service_name
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) >= 2:
                            # Parse the output: PID, Status, Label
                            parts = lines[1].split('\t')
                            if len(parts) >= 3:
                                pid_str = parts[0].strip()
                                status_str = parts[1].strip()
                                
                                if pid_str != '-':
                                    status['pid'] = int(pid_str)
                                    status['running'] = True
                                
                                if status_str != '-':
                                    status['exit_code'] = int(status_str)
                
                except Exception as e:
                    logger.error(f"Error parsing service status: {e}")
        
        return status
    
    def restart_service(self) -> bool:
        """Restart the service"""
        try:
            if self.is_service_loaded():
                # Unload first
                subprocess.run([
                    'launchctl', 'unload', self.plist_path
                ], capture_output=True, text=True)
            
            # Load again
            result = subprocess.run([
                'launchctl', 'load', self.plist_path
            ], capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to restart service: {e}")
            return False


def get_service_manager() -> MacOSServiceManager:
    """Get the service manager instance"""
    return MacOSServiceManager()