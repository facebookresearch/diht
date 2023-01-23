# DiHT
"Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training" [[Paper](https://arxiv.org/abs/2301.02280)]

## Installation

```bash
git clone https://github.com/facebookresearch/diht
cd diht
pip install -r requirements.txt
pip install -e .  # To install as an editable package
```

You can use `pip install .` to install the codebase as a non-editable package.

## Example

```python
import torch

from diht import model_zoo
from PIL import Image


text_tokenizer, image_transform, model = model_zoo.load_model(
    "diht_vitl14_336px", is_train=False
)

image = Image.open("infer_image.png").convert("RGB")
image = image_transform(image).unsqueeze(0)
text_captions = ["a mountain", "a beach", "a desert"]
text = text_tokenizer(text_captions)

with torch.no_grad():
    image_features, text_features, logit_scale = model(image, text)
    logits_per_image = logit_scale * image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).numpy()

print(f"text captions: {text_captions}")
print(f"text caption probs: {probs}")
```

The above code snippet should output
```bash
text captions: ['a mountain', 'a beach', 'a desert']
text caption probs: [[0.99370664 0.00514017 0.00115326]]
```

## Running on GPU/CPU

By default the model runs on CPU, to run on GPU you can do `model = model.to(torch.device("cuda"))`. The image and text tensors will also have to be transferred accordingly.

## Available Models

```python
import diht


print(diht.available_models())
```

## Citation
If you find this model useful, please consider citing our preprint using the citation below.
``` 
@article{rdk+23,
  title = {Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training},
  author = {Radenovic, Filip and Dubey, Abhimanyu and Kadian, Abhishek and Mihaylov, Todor and Vandenhende, Simon and Patel, Yash and Wen, Yi and Ramanathan, Vignesh and Mahajan, Dhruv},
  journal = {arXiv:2301.02280},
  year = {2023}
}
```

## License
```
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
```
