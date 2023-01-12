# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from diht.model import DiHT
from diht.model_zoo import available_models, load_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


__all__ = [
    "available_models",
    "load_model",
    "DiHT",
]
