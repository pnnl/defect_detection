"""Install setup.py for defect_detection."""
from setuptools import setup, find_packages

def requirements(filename):
    with open(filename, "r") as stream:
        return stream.read().split("\n")

setup(
    name='defect_detection',
    version='0.1',
    description='TTP Defect Detection',
    author='Karl Pazdernik',
    author_email='karl.pazdernik@pnnl.gov',
    url='na',
    long_description=open('README.md').read(),
    packages=find_packages(),
    scripts=['bin/defect_detector', 'bin/defect_trainer'],
    install_requires=[
        "pandas==1.4.3",
        "numpy==1.26.4",
        "matplotlib==3.9.0",
        "torch==1.10.1",
        "torchvision==0.11.2",
        "scikit-learn==1.1.1",
        "tifffile==2024.4.18",
        "tqdm==4.62.3",
        "IPython==8.14.0",
        "statsmodels==0.13.1",
        "opencv-contrib-python-headless==4.5.4.60",
        "gudhi==3.5.0",
        "Pillow==9.2.0",
        "pytest==8.3.3",
        "comm>=0.1.1"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ]
)
