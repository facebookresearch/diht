# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch

from diht import model_zoo
from PIL import Image


class TestDiHT(unittest.TestCase):
    def test_model(self):
        text_tokenizer, image_transform, model = model_zoo.load_model(
            "diht_vitb16_224px", is_train=False
        )
        image = Image.open("infer_image.png").convert("RGB")
        image = image_transform(image).unsqueeze(0)
        text_captions = ["a mountain", "a beach", "a desert"]
        text = text_tokenizer(text_captions)

        with torch.no_grad():
            image_features, text_features, logit_scale = model(image, text)
            logits_per_image = logit_scale * image_features @ text_features.T
            _ = logits_per_image.softmax(dim=-1).numpy()
