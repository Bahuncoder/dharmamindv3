"""
🕉️ DharmaMind Vision Setup Script

Traditional Yoga Computer Vision System setup for development and distribution.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dharmamind-vision",
    version="1.0.0",
    author="DharmaMind AI Team",
    author_email="team@dharmamind.ai",
    description="Traditional Yoga Computer Vision System for Pose Detection and Spiritual Guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dharmamind/dharmamind-vision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Religion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "tensorflow": [
            "tensorflow>=2.13.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "onnx": [
            "onnxruntime>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dharmamind-vision=dharmamind_vision.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dharmamind_vision": [
            "models/*.pkl",
            "models/*.joblib",
            "data/*.json",
            "config/*.yaml",
        ],
    },
    keywords=[
        "yoga", "computer-vision", "pose-detection", "meditation", "spiritual-ai",
        "traditional-yoga", "hatha-yoga", "asana-classification", "alignment",
        "chakra", "mediapipe", "fastapi", "machine-learning", "dharma"
    ],
    project_urls={
        "Bug Reports": "https://github.com/dharmamind/dharmamind-vision/issues",
        "Source": "https://github.com/dharmamind/dharmamind-vision",
        "Documentation": "https://dharmamind-vision.readthedocs.io/",
        "Funding": "https://github.com/sponsors/dharmamind",
    },
)