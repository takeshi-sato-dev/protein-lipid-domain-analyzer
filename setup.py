from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lipid-domain-transport-analyzer",
    version="1.0.0",
    author="Takeshi Sato",
    author_email="tsato@kyoyaku.ac.jp",
    description="A configurable toolkit for studying membrane lipid-mediated protein transport using Hidden Markov Models and Bayesian statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "MDAnalysis>=2.0.0",
        "pymc>=5.0.0",
        "arviz>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "lipid-domain-analyzer=main:main",
        ],
    },
    keywords="molecular dynamics, membrane biology, lipid transport, Hidden Markov Models, Bayesian statistics, cholesterol domains, gangliosides",
    project_urls={
        "Bug Reports": "https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer/issues",
        "Source": "https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer",
        "Documentation": "https://github.com/tsato-kyoyaku/lipid-domain-transport-analyzer/README.md",
    },
)