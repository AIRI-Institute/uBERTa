# Notebooks

This folder contains notebooks sufficient to fully reproduce the study.

Although each notebook is self-contained, we suggest the following order.

1. Prepare the data in `prepare_base_dataset_v4.7.ipynb`.
2. Train `distilBERT` on the TIS classification objective in `train_uBERTa.ipynb`.
3. Use the trained model to obtain predictions of TISs on 5'UTR sequences of the human genome in `predict_5UTR.ipynb`
4. Compare predictions with those published previously in `compare_predictions.ipynb`.

---
Obsolete and given solely for completeness: `pretrain_distil.ipynb`, `prepare_base_dataset.ipynb`

---
XGBoost model has its own dependencies. Please refer to the `xgb.ipynb` for further information.

---

Each notebook has prerequisites which one can obtain following the links. 
In CLI, they can be downloaded using [gdown](https://github.com/wkentaro/gdown) (usually with `--fuzzy` argument).