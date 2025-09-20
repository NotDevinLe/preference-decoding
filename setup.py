from setuptools import setup, find_packages

setup(
    name="preference_decoding",
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
        "fastapi",
        "uvicorn",
        "pydantic",
        "aiohttp",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "gumbel-collector=src.core.collector_server:main",
            "gumbel-learner=src.core.learner_server:main",
            "gumbel-coordinator=src.core.coordinator:main",
            "gumbel-test-collector=tests.test_collector:main",
        ],
    },
    author="Devin",
    author_email="devin.t.le@outlook.com",
)