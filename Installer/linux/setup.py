from setuptools import setup, find_packages
import os
import shutil

# Read the README file
current_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_dir, "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read version from VERSION file and copy to package directory
def read_version():
    # Single source of truth: repository root VERSION file
    repo_version_file = os.path.join(current_dir, '..', '..', 'VERSION')
    pkg_version_file = os.path.join(current_dir, 'kwaainet', 'VERSION')

    if os.path.exists(repo_version_file):
        with open(repo_version_file, 'r') as f:
            version = f.read().strip()

        # Copy VERSION to package directory to maintain single source of truth
        os.makedirs(os.path.dirname(pkg_version_file), exist_ok=True)
        shutil.copy2(repo_version_file, pkg_version_file)
        print(f"Synced VERSION file: {version}")

        return version
    return "0.0.0"  # Fallback version

VERSION = read_version()

setup(
    name="kwaainet-linux",
    version=VERSION,
    author="Kwaai Labs",
    author_email="contact@kwaai.ai",
    description="KwaaiNet for Linux - Native compute sharing with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kwaai-AI-Lab/OpenAI-Petal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "PyYAML>=6.0.2",  # Security fixes
        # NOTE: petals is installed separately by the installer from source (2.3.0.dev2)
        # "petals>=2.3.0" would fail since PyPI only has up to 2.2.0
        "torch>=1.12.0,<2.4.0",  # Compatible with Petals and hivemind - tested range
        "transformers>=4.32.0,<4.45.0",  # Petals 2.3.0.dev2 requires 4.43.1
        "accelerate>=0.20.0",  # Broader compatibility
        "requests>=2.28.0",  # Security fixes with broader compatibility
        "tqdm>=4.64.0",  # Stable version
        "psutil>=5.9.0",  # Process and system monitoring for daemon functionality
        "huggingface_hub>=0.16.4",  # Compatible with transformers (relaxed from >=0.34.0)
        "tokenizers>=0.14.0,<0.20.0",  # Petals 2.3.0.dev2 requires 0.19.1
        "py-multihash<2.0",  # Hivemind 1.1.11 requires FuncReg API (not in 2.0+)
        "triton<3.0",  # bitsandbytes 0.41.1 requires triton.ops module (removed in 3.0+)
    ],
    extras_require={
        "cuda": [
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kwaainet=kwaainet.runner:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)