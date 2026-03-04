from setuptools import find_packages, setup

setup(
    name="tem-psd-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "matplotlib",
        "seaborn",
        "pandas",
        "click",
        "tqdm",
        "ncempy",
        "Pillow",
    ],
    entry_points={"console_scripts": ["tem-psd=tem_psd.cli:cli"]},
)
