from setuptools import setup, find_packages

setup(
    name="neuro_mend",
    version="1.0.0",
    author="Mostafa Shams",
    description="A Closed-Loop Active Learning Framework for Auto-Healing Vision Models",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.25.0",
        "transformers>=4.38.0",
        "accelerate>=0.26.0",
        "umap-learn",
        "hdbscan",
        "scikit-learn",
        "matplotlib",
        "pillow",
        "opencv-python"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.10',
)
