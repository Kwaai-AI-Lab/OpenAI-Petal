from setuptools import setup, find_packages
import os
import platform

# Check if running on macOS
if platform.system() != "Darwin":
    print("WARNING: This package is specifically designed for macOS systems.")

# Read version from VERSION file
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return "0.0.0"  # Fallback version

VERSION = read_version()

# Define dependencies - no CUDA packages as they're not needed on macOS
dependencies = [
    "torch>=2.0.0,<2.4.0",  # Tested range for Petals and hivemind
    "peft>=0.6.0",  # Match the Docker version (--no-deps)
    "petals @ git+https://github.com/bigscience-workshop/petals",
    "transformers>=4.32.0,<4.35.0",  # Petals 2.2.0 compatibility (required by current petals)
    "pyarrow>=10.0.0",  # Security updates
    "requests>=2.32.0",  # Security fixes
    "tqdm>=4.66.0",  # Latest stable
    "pyyaml>=6.0.0",  # Security fixes
    "psutil>=5.9.0",  # System monitoring with security updates
    "accelerate>=0.20.0",  # ML acceleration library
    "py-multihash<2.0",  # Hivemind 1.1.11 requires FuncReg API (not in 2.0+)
]

# Add any Mac-specific dependencies
if platform.system() == "Darwin":
    # Pin bitsandbytes to match Petals 2.3.0.dev2 requirement
    dependencies.append("bitsandbytes==0.41.1")

setup(
    name="kwaainet-mac",
    version=VERSION,
    description="KwaaiNet Node for Mac systems to share compute/GPU resources",
    author="Kwaai Labs",
    author_email="contact@kwaai.ai",
    url="https://github.com/yourusername/kwaainet-mac",
    packages=find_packages(),
    install_requires=dependencies,
    scripts=["scripts/kwaainet-start"],
    python_requires=">=3.10,<3.11",
    entry_points={
        "console_scripts": [
            "kwaainet=kwaainet.runner:main",
        ],
    },
    package_data={
        'kwaainet': ['VERSION'],  # Include VERSION file in package
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS",
    ],
    include_package_data=True,
)