# uBERTa

uORF BERT model to annotate putative TISs of the human genome.

----

# Installation

0. (Optional) Create a new conda environment, e.g. 
   1. `conda create -n uberta python=3.8 -c conda-forge -y`
   2. `conda activate uberta`
1. Clone this repo `git clone https://github.com/skoblov-lab/uBERTa.git`
2. `cd uBERTa`
3. Install dependencies
   1. Install [pytorch](https://pytorch.org/get-started/locally/), e.g., `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
   2. Install `uBERTa` dependencies `pip install setup.py`

To go through the notebooks, additional dependencies are required. 
We assume you are using conda.

- Jupyter (e.g., jupyter lab): `conda install jupyterlab -c conda-forge -y`
- `conda install -c conda-forge -y scikit-learn scipy`
- `pip install pyliftover`

# Usage

Use [this link](https://drive.google.com/file/d/1weL5Wp3DrCIoW-kCxJ6aIveYZyQQbDOQ/view?usp=sharing) to download the trained model.

Currently, for proper functioning, uBERTa requires an experimental signal as a part of the input. 
Thus, its usage is limited to the regions of the human genomes well covered by experimental data. 
Keep this in mind while utilizing the model. 
The basic usage example is provided within `notebooks/basic_usage.ipynb`. 
For more advanced usage, consider exploring `predict_5UTR.ipynb`.

# Note on XGBoost

XGBoost demonstrated better performance than the distilBERT model as explained in the paper.
`notebooks/xgb.ipynb` contains all the code to train, validate, and predict 5'UTR TISs.
The trained XGBoost model is available via [this link](https://drive.google.com/file/d/1gWby2I95rFf0AzLFWxCfWqztLM-lCW5U/view?usp=sharing).

# Predictions

To download predictions for 5'UTRs, please use the following links:
- [uBERTa predictions](https://drive.google.com/file/d/1I3B2TVX1Xu8MdfEOJaIJqgctRgWnwdPD/view?usp=sharing)
- [XGBoost predictions](https://drive.google.com/file/d/1fYrjqYB9CXwX5_-IqoQwZB2tBx3Yk8Jt/view?usp=sharing)

The archives contain:

- `predictions_5UTR.tsv` table with predictions and prediction probabilities for 5'UTRs. 
Genomic positions are given from the start-codon's first nucleotide, which is reversed for the "-" strand.
- `predictions_5UTR.bed` file that can be loaded into genomic browser. 
Prediction probabilities are given in percents in the fifth column, while prediction types are denoted by colors and depend on the dataset.
Namely, for the inference dataset encompassing 5'UTRs that did not undergo manual curation, green and blue colors denote positive and negative predictions.
For 5'UTRs that did undergo manual curation, the colors are the following:
  - green -- True Positive (TP)
  - blue -- True Negative (TN)
  - red -- False Negative (FN)
  - black -- False Positive (FP)
- `prediction_scores.tsv` table with prediction scores per dataset and start codon.


# Additional links

Check https://github.com/bioinf/uORF_annotator for uORF_annotator -- a tool to annotate functional impacts of the discovered uORFs.