from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_dir, "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="kwaainet-linux",
    version="0.2.2",
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
        "petals>=2.2.0",  # Latest stable (requires transformers<4.35.0)
        "torch>=1.12.0",  # Broader compatibility for older Python
        'transformers>=4.32.0,<4.35.0; python_version=="3.7"',  # Python 3.7 compatible (Petals constraint)
        'transformers>=4.32.0,<4.35.0; python_version>="3.8"',  # Petals compatibility constraint (security compromise)
        "accelerate>=0.20.0",  # Broader compatibility
        "requests>=2.28.0",  # Security fixes with broader compatibility
        "tqdm>=4.64.0",  # Stable version
        "psutil>=5.9.0",  # Process and system monitoring for daemon functionality
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
            "kwaainet-linux=kwaainet.runner:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)