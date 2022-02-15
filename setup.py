from distutils.core import setup
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='uBERTa',
    version='0.1',
    author='Ivan Reveguk',
    author_email='ivan.reveguk@gmail.com',
    # description='',
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
    install_requires=[
        'pysam>=0.17.0',
        'setuptools>=58.0.4',
        'pandas>=1.4.0',
        'numpy>=1.21.2',
        'biopython>=1.79',
        'tqdm>=4.62.3',
        'toolz>=0.11.2',
        'more-itertools>=8.12.0',
        'transformers>=4.15.0',
        'pytorch-lightning>=1.5.7',
        'tables>=3.6.1',
        'pyBigWig>=0.3.18'
    ],
)
