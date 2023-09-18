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
model = MolDetect(ckpt_path, device=torch.device('cpu'))

text = "The chemical formula of water is H2O"
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

## Train and Evaluate ChemNER



