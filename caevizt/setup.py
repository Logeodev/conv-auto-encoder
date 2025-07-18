from setuptools import setup, find_packages

setup(
    name="caevizt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.10.0",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "scikeras==0.12.0"
    ],
)