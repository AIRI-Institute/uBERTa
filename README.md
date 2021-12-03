# uBERTa

----

## `DataGenerator` usage

Prerequisites:
1. Clone this repo
2. Run `pip install -e /path/to/repo`
3. [Download](https://drive.google.com/file/d/1ttcKSt9I5viiZendOV7_36HBtmClbmZd/view?usp=sharing) hg38 assembly.
4. [Download](https://drive.google.com/file/d/1keUHHHeM_97owdOP-wUpuP7gBO-_lFfb/view?usp=sharing) our "base" dataset.

Remarks:
- You may need to unpack the base dataset first due to pandas having header parsing issues of the archived table (e.g., `tar -xzf DS_BASE.tsv.tar.gz`)
- For CLI downloads consider using [gdown](https://github.com/wkentaro/gdown) as `gdown --fuzzy ...` with the links above.

Follow-up: check the Jupyter notebook `generator_usage.ipynb`
