"""
Setup script for pointcept_classifier package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pointcept_classifier",
    version="0.1.0",
    author="IceCube Collaboration",
    description="Pointcept-based event classifier for IceCube neutrino detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/briedel/pointcept_classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "PyYAML>=5.4.0",
        "h5py>=3.0.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "icecube-train=train:main",
            "icecube-infer=inference:main",
        ],
    },
)
