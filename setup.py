from setuptools import find_packages, setup

setup(
    name="shadowweave",
    version="0.2.0",
    description="Shadow-ray navigation via spatial audio",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # default.yaml must ship with the package or load_config() breaks on an
    # installed (non-editable) copy.
    package_data={"shadowweave": ["configs/*.yaml"]},
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "omegaconf>=2.3.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "sim": ["mujoco>=3.0.0", "gymnasium>=0.29.0", "stable-baselines3>=2.2.0"],
        "viz": ["plotly>=5.18.0", "matplotlib>=3.7.0", "gradio>=4.0.0"],
        "audio": ["sounddevice>=0.4.6"],
        "depth": ["transformers>=4.40.0", "Pillow>=10.0.0"],
        "dev": ["pytest>=7.4.0", "wandb>=0.16.0", "tensorboard>=2.15.0"],
    },
)
