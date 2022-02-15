# uBERTa

uORF BERT model to annotate putative TIS of the human genome.

----

## Installation

0. (Optional) Create a new conda environment, e.g. 
   1. `conda create -n uberta python=3.8 -c conda-forge -y`
   2. `conda activate uberta`
1. Clone this repo `git clone https://github.com/skoblov-lab/uBERTa.git`
2. Install dependencies
   1. Install [pytorch](https://pytorch.org/get-started/locally/), e.g., `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
   2. Install `uBERTa` dependencies `pip install -e ./uBERTa`

To go through the notebooks, additional dependencies are required. 
We assume you are using conda.

- Jupyter (e.g., jupyter lab): `conda install -c conda-forge -y jupyterlab ipywidgets`
- `conda install -c conda-forge -y scikit-learn scipy`
- `pip install pyliftover`

## Usage

Currently, for proper functioning, uBERTa requires an experimental signal as a part of the input.
Thus, its usage is limited to the regions of the human genomes well covered by experimental data.
Keep this in mind while utilizing the model.
The basic usage example is provided in `notebooks/basic_usage.ipynb`.
For more advanced usage, consider exploring `predict_5UTR.ipynb`.
Predictions are available via [this link](https://drive.google.com/file/d/1a6JlTgCKxO45nRLJ1qMpi6B-8OYLCrWf/view?usp=sharing).
