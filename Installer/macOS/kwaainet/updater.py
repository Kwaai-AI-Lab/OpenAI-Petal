"""
Auto-update functionality for KwaaiNet
Checks for new versions and manages updates
"""

import os
import sys
import json
import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# GitHub repository information
GITHUB_OWNER = "Kwaai-AI-Lab"
GITHUB_REPO = "OpenAI-Petal"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
VERSION_FILE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/VERSION"


class UpdateChecker:
    """Checks for available updates and manages version information"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.expanduser("~/.kwaainet")
        self.cache_file = os.path.join(self.data_dir, "update_cache.json")
        self.cache_ttl = 3600  # 1 hour cache

        # Get current version
        self.current_version = self._get_current_version()

    def _get_current_version(self) -> str:
        """Get the currently installed version"""
        try:
            # Try to read from installed package
            import kwaainet
            if hasattr(kwaainet, '__version__'):
                return kwaainet.__version__
        except (ImportError, AttributeError):
            pass

        # Fallback: read from VERSION file in installation directory
        try:
            version_file = os.path.join(os.path.dirname(__file__), '../../VERSION')
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.debug(f"Could not read VERSION file: {e}")

        return "unknown"

    def _read_cache(self) -> Optional[Dict]:
        """Read cached version information"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)

                # Check if cache is still valid
                cached_time = cache.get('timestamp', 0)
                if time.time() - cached_time < self.cache_ttl:
                    return cache
        except Exception as e:
            logger.debug(f"Failed to read cache: {e}")

        return None

    def _write_cache(self, data: Dict):
        """Write version information to cache"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            cache = {
                **data,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write cache: {e}")

    def _fetch_latest_version_from_api(self) -> Optional[Dict]:
        """Fetch latest version from GitHub Releases API"""
        try:
            import requests

            response = requests.get(GITHUB_API_URL, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return {
                    'version': data.get('tag_name', '').lstrip('v'),
                    'name': data.get('name', ''),
                    'url': data.get('html_url', ''),
                    'body': data.get('body', ''),
                    'published_at': data.get('published_at', '')
                }
            else:
                logger.debug(f"GitHub API returned status {response.status_code}")

        except ImportError:
            logger.debug("requests module not available for API check")
        except Exception as e:
            logger.debug(f"Failed to fetch from GitHub API: {e}")

        return None

    def _fetch_latest_version_from_file(self) -> Optional[str]:
        """Fetch latest version from VERSION file on GitHub"""
        try:
            import requests

            # Add cache-busting parameter to avoid GitHub CDN cache
            cache_bust = f"?t={int(time.time())}"
            url = VERSION_FILE_URL + cache_bust

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.text.strip()
            else:
                logger.debug(f"VERSION file fetch returned status {response.status_code}")

        except ImportError:
            logger.debug("requests module not available")
        except Exception as e:
            logger.debug(f"Failed to fetch VERSION file: {e}")

        return None

    def check_for_updates(self, force: bool = False) -> Optional[Dict]:
        """
        Check if a new version is available

        Returns:
            Dict with update info if available, None if up-to-date or check failed
        """
        # Check cache first unless forced
        if not force:
            cached = self._read_cache()
            if cached:
                logger.debug("Using cached version information")
                # Still validate against current version
                if self._compare_versions(cached.get('version', ''), self.current_version) > 0:
                    return cached
                return None

        # Try both GitHub Releases API and VERSION file
        latest_info = self._fetch_latest_version_from_api()
        version_file = self._fetch_latest_version_from_file()

        # Use whichever version is higher (or VERSION file if API fails)
        if latest_info and version_file:
            api_version = latest_info.get('version', '')
            if self._compare_versions(version_file, api_version) > 0:
                # VERSION file has newer version than latest release
                latest_info = {
                    'version': version_file,
                    'url': f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tag/v{version_file}"
                }
        elif version_file and not latest_info:
            # API failed, use VERSION file
            latest_info = {
                'version': version_file,
                'url': f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tag/v{version_file}"
            }

        if not latest_info:
            logger.debug("Could not fetch latest version information")
            return None

        # Cache the result
        self._write_cache(latest_info)

        # Compare versions
        latest_version = latest_info.get('version', '')
        if self._compare_versions(latest_version, self.current_version) > 0:
            return latest_info

        return None

    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two semantic version strings

        Returns:
            1 if v1 > v2
            0 if v1 == v2
            -1 if v1 < v2
        """
        try:
            # Remove 'v' prefix if present
            v1 = v1.lstrip('v')
            v2 = v2.lstrip('v')

            # Split into parts and convert to integers
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]

            # Pad with zeros if needed
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))

            # Compare
            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1

            return 0

        except Exception as e:
            logger.debug(f"Version comparison failed: {e}")
            return 0


class Updater:
    """Manages the update process"""

    def __init__(self):
        self.checker = UpdateChecker()
        self.backup_dir = os.path.expanduser("~/.kwaainet/backups")

    def _detect_installation_method(self) -> str:
        """
        Detect how KwaaiNet was installed

        Returns:
            'git' if installed from git repository
            'installer' if installed via installer script
            'pip' if installed via pip
            'unknown' if cannot determine
        """
        try:
            import kwaainet
            install_path = os.path.dirname(kwaainet.__file__)

            # Check if in a git repository
            git_dir = os.path.join(install_path, '../../.git')
            if os.path.exists(git_dir):
                return 'git'

            # Check if installed via installer (editable install from ~/.kwaainet/source)
            if '.kwaainet/source' in install_path:
                return 'installer'

            # Default to pip
            return 'pip'

        except Exception as e:
            logger.debug(f"Could not detect installation method: {e}")
            return 'unknown'

    def _backup_config(self) -> Optional[str]:
        """Backup current configuration"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)

            config_file = os.path.expanduser("~/.kwaainet/config.yaml")
            if os.path.exists(config_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(self.backup_dir, f"config_{timestamp}.yaml")
                shutil.copy2(config_file, backup_file)
                logger.info(f"Backed up config to {backup_file}")
                return backup_file

        except Exception as e:
            logger.error(f"Failed to backup config: {e}")

        return None

    def update(self, target_version: Optional[str] = None) -> bool:
        """
        Perform the update

        Args:
            target_version: Specific version to update to (None = latest)

        Returns:
            True if update successful, False otherwise
        """
        # Check what's available
        update_info = self.checker.check_for_updates(force=True)

        if not update_info and not target_version:
            logger.info("Already on the latest version")
            return True

        latest_version = update_info.get('version') if update_info else target_version

        logger.info(f"Updating from {self.checker.current_version} to {latest_version}")

        # Backup configuration
        backup = self._backup_config()
        if backup:
            logger.info(f"✅ Configuration backed up")

        # Detect installation method
        install_method = self._detect_installation_method()
        logger.info(f"Installation method: {install_method}")

        # Perform update based on method
        if install_method == 'git':
            return self._update_git()
        elif install_method == 'installer':
            return self._update_installer()
        else:
            return self._update_pip()

    def _update_git(self) -> bool:
        """Update via git pull"""
        try:
            import kwaainet
            repo_path = os.path.join(os.path.dirname(kwaainet.__file__), '../..')

            logger.info("Updating via git pull...")

            # Git pull
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Git pull failed: {result.stderr}")
                return False

            logger.info("✅ Git pull successful")

            # Reinstall package
            logger.info("Reinstalling package...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'],
                cwd=os.path.join(repo_path, 'Installer/macOS'),
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Package reinstall failed: {result.stderr}")
                return False

            logger.info("✅ Update complete!")
            return True

        except Exception as e:
            logger.error(f"Git update failed: {e}")
            return False

    def _update_installer(self) -> bool:
        """Update by re-running installer"""
        logger.info("Updating via installer re-run...")
        logger.info("Downloading latest installer...")

        # This would download and run the latest installer
        # For now, just show instructions
        logger.info("Please run:")
        logger.info("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macOS/macinstaller.sh)\"")

        return False

    def _update_pip(self) -> bool:
        """Update via pip install"""
        try:
            logger.info("Updating via pip...")

            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade',
                 'git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/macOS'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Pip update failed: {result.stderr}")
                return False

            logger.info("✅ Update complete!")
            return True

        except Exception as e:
            logger.error(f"Pip update failed: {e}")
            return False
