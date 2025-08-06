from setuptools import setup, find_packages

setup(
    name="preference-decoding",
    version="0.1.0",
    description="Preference learning and decoding research project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "transformers",
        "vllm",
        "wandb",
        "tqdm",
        "datasets",
    ],
    author="Devin",
    author_email="devin.t.le@outlook.com",
)