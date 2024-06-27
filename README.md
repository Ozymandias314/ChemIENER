# ChemIENER
This is the repository for ChemNER, a named entity recognition model for chemical entities used in [`OpenChemIE`](mit.openchemie.info).

## Quick Start
Run the following command to install the package and its dependencies:
```
git clone git@github.com:Ozymandias314/ChemIENER.git
cd chemiener
python setup.py install
```

Download the checkpoint and use ChemNER to extract chemical entities from chemical descriptions:

```python 
import torch
from chemiener import ChemNER
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("Ozymandias314/ChemNERCkpt", "best.ckpt")
model = ChemNER(ckpt_path, device=torch.device('cpu'))

text = ["The chemical formula of water is H2O"]
predictions = model.predict_image_file(text)
```
The predictions are given in character-level spans, and have the following format:
```python
[
  (
    CATEGORY_NAME #string,
    [span_start, span_end] #input[span_start:span_end] is a detected entity of category CATEGORY_NAME
  ),
  #more predictions
]
```

## Data
We train our model on the [`CHEMDNER`](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2) corpus of annotated titles and abstracts from chemical literature. The processed
data can be found at [`data/`](data).

## Train and Evaluate ChemNER
Run this script to train and evaluate ChemNER.
```bash
bash scripts/train_ner.sh
```



