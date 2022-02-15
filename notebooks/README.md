# Notebooks

This folder contains notebooks sufficient to fully reproduce the study.

Although each notebook is self-contained, we suggest the following order.

1. Prepare the data in `prepare_base_dataset.ipynb`.
2. Pretrain the `DistilBert` model on the MLM objective to serve as our "core" transformer block in `pretrain_distil.ipynb`.
3. Fine-tune the pretrained model on the TIS classification objective in `train_uBERTa.ipynb`.
4. Use the fine-tuned model to obtain predictions of TISs on 5'UTR sequences of the human genome in `predict_5UTR.ipynb`
5. Compare predictions with those published previously in `compare_predictions.ipynb`.

---

Each notebook has prerequisites which one can obtain following the links. 
In CLI, they can be downloaded using [gdown](https://github.com/wkentaro/gdown) (usually with `--fuzzy` argument).