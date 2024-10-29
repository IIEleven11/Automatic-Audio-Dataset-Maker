from setuptools import setup, find_packages

setup(
    name="dataspeech",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "datasets[audio]",
        "penn",
        "g2p",
        "demucs",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "torch",
        "torchaudio",
        "numpy",
        "multiprocess",
    ],
    dependency_links=[
        "https://github.com/marianne-m/brouhaha-vad/archive/main.zip#egg=brouhaha-vad"
    ]
)