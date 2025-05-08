from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read development requirements from requirements-dev.txt
with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    dev_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="jaxabm",
    version="0.1.0",
    author="Duy Pham",
    author_email="duypham@stanford.edu",
    description="A JAX-accelerated agent-based modeling framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duypham/JaxABM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
)
