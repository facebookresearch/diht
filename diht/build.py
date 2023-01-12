# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
from fvcore.common.registry import Registry


MODELS_REGISTRY = Registry("MODELS")
MODELS_REGISTRY.__doc__ = """
Registry for models.
Registered object must return instance of :class: `torch.nn.Module`
"""


def build_model(model_name, *args, **kwargs):
    model = MODELS_REGISTRY.get(model_name)(*args, **kwargs)
    assert isinstance(model, nn.Module)
    return model
