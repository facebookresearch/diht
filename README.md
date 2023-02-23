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

## Example ImageNet-1K zero-shot evaluation

A simple image classification zero-shot evaluation using a single GPU can be performed by running:
> **Note**: Download ImageNet-1K dataset from the original website. Edit `IMAGENET_ROOT` in `example_imagenet_eval.py` to match the location on your machine.
```
python example_imagenet_eval.py
```
For DiHT-L/14@336 the output should look like:
```
ImageNet1K acc@1 for diht_vitl14_336px: 77.9
```

## Example retrieval zero-shot evaluation

A simple retrieval zero-shot evaluation using a single GPU can be performed by running:
> **Note**: Download COCO and Flickr30K datasets from the original websites. Json files (`coco_test.json` and `flickr30k_test.json`) can be downloaded from https://github.com/salesforce/ALBEF#download. Edit `COCO_ROOT` and `FLICKR30K_ROOT` in `example_retrieval_eval.py` to match the locations on your machine.
```
python example_retrieval_eval.py
```
For DiHT-L/14@336 the output should look like:
```
COCO T2I r@1 for diht_vitl14_336px: 49.3
COCO I2T for diht_vitl14_336px: 65.3

Flickr30K T2I r@1 for diht_vitl14_336px: 78.2
Flickr30K I2T for diht_vitl14_336px: 91.1
```

## Zero-shot model performance

| Model | ImageNet-1K | COCO T2I | COCO I2T | Flickr30K T2I | Flickr30K I2T |
| :---  |    :----:   |  :----:  |  :----:  |     :----:    |     :----:    |
|       |  Accuracy@1 | Recall@1 | Recall@1 |    Recall@1   |    Recall@1   |
| diht_vitb32_224px      | 68.0 | 40.6 | 59.3 | 68.6 | 84.4 |
| diht_vitb16_224px      | 72.2 | 43.3 | 60.3 | 72.9 | 89.8 |
| diht_vitl14_224px      | 77.0 | 48.0 | 65.1 | 76.7 | 92.0 |
| diht_vitl14_336px      | 77.9 | 49.3 | 65.3 | 78.2 | 91.1 |


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
