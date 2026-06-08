from setuptools import setup, find_packages

setup(
    name="shadowweave",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "omegaconf>=2.3.0",
        "scipy>=1.10.0",
    ],
)
