from setuptools import setup, find_packages
import os
import platform

# Check if running on macOS
if platform.system() != "Darwin":
    print("WARNING: This package is specifically designed for macOS systems.")

# Define dependencies - no CUDA packages as they're not needed on macOS
dependencies = [
    "torch>=1.12",  # Match the Docker version requirement
    "peft>=0.6.0",  # Match the Docker version (--no-deps)
    "petals @ git+https://github.com/bigscience-workshop/petals",
    "pyarrow",  # Often needed for data handling
    "requests",
    "tqdm",
    "pyyaml",
    "psutil",  # For system monitoring
]

# Add any Mac-specific dependencies
if platform.system() == "Darwin":
    # Check if M1/M2 Mac for specific optimizations
    if platform.processor() == 'arm':
        # For Apple Silicon Macs - use specific bitsandbytes version
        dependencies.append("bitsandbytes")
    else:
        # For Intel Macs
        dependencies.append("bitsandbytes")

setup(
    name="kwaainet-mac",
    version="0.7.0",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS",
    ],
    include_package_data=True,
)