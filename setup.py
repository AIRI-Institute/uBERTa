from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='uBERTa',
    version='0.1dev1',
    author='Ivan Reveguk',
    author_email='ivan.reveguk@gmail.com',
    # description='',
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pysam>=0.17.0',
        'setuptools>=58.0.4',
        'pandas',
        'numpy',
        'biopython',
        'tqdm',
        'toolz',
        'more-itertools>=8.12.0',
        'networkx',
        'ncls',
        'scipy',
        'transformers',
#        'pytorch',
        'pytorch-lightning'
    ],
    # include_package_data=True,
    # package_data={
    #     '': ['*.conf', '*.exe']
    # },
    # packages=find_packages(exclude=['test']),
)
