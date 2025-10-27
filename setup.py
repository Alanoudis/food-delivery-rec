# setup.py
from setuptools import setup, find_packages

setup(
    name="food-delivery-rec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "implicit>=0.7.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
)
