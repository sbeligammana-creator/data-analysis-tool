#!/usr/bin/env python
"""Setup configuration for data_analysis package."""

from setuptools import setup, find_packages

setup(
    name="data-analysis",
    version="0.1.0",
    description="A utility package for data analysis with inspection and plotting methods",
    long_description=open("README.md", encoding="utf-8").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/data-analysis",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="data-analysis data-inspection plotting",
    zip_safe=False,
)
