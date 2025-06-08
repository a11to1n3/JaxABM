from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jaxabm",
    version="0.1.5",
    author="Anh-Duy Pham, Paola D'Orazio",
    author_email="duyanhpham@outlook.com",
    description="A JAX-accelerated agent-based modeling framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a11to1n3/JaxABM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "polars>=0.19.3",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "networkx>=2.6.0",
        "tqdm>=4.62.0",
        "pyDOE2>=1.3.0",
        "SALib>=1.4.5",
        "jax>=0.4.1",
        "jaxlib>=0.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=21.0.0",
            "mypy>=0.800",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
) 