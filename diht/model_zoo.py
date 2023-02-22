# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Functionalities borrowed from openai/CLIP
# ref: https://github.com/openai/CLIP/blob/main/clip/clip.py


import hashlib
import logging
import os
import urllib
from pathlib import Path

import torch

from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from diht.build import build_model
from diht.tokenizer import ClipTokenizer
from diht.transforms import image_transform

CONFIG_DIRPATH = Path(__file__).parent.joinpath("model_zoo_configs")


def _get_sha_from_url(checkpoint_url):
    return checkpoint_url.split("_")[-1][: -len(".ckpt")]


def _download_model_ckpt(checkpoint_url, root):
    root_path = Path(root)
    root_path.mkdir(exist_ok=True)
    filename = os.path.basename(checkpoint_url)

    expected_sha256 = _get_sha_from_url(checkpoint_url)
    download_target = root_path.joinpath(filename)

    if download_target.exists() is True:
        if download_target.is_file() is False:
            raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            logging.info(f"using the already existing checkpoint at {download_target}")
            return str(download_target)
        else:
            logging.warning(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(checkpoint_url) as source, open(
        download_target, "wb"
    ) as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return str(download_target)


def _load_model_ckpt(model, checkpoint_path):
    checkpoint_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_state_dict)
    return model


def available_models():
    return sorted(
        [model.name[: -len(".yaml")] for model in CONFIG_DIRPATH.rglob("*.yaml")]
    )


def load_model(model_name, is_train=False, download_root=None):
    """
    Load a model from the model zoo.

    Input:
    ------
    model_name: str
        Name of the model to load. Support values are `vitb16_224px_ozi_sota`,
        `vitb32_224px_ozi_sota` and `vitl14_336px_ozi_sota`.
    is_train: bool
        Whether to return the model in train or eval mode. Note, if `is_train=True`
        the image transform will be the one used during training, and not evaluation.

    Return:
    -------
    text_tokenizer: Callable
        Tokenizer to be used with the vision-language model.
    image_transform: Callable
        Image transformation to be used with the vision-language model.
    model: nn.Module
        The actual pytorch module which implements `encode_text` and `encode_image`
        functions.
    """
    config_filepath = CONFIG_DIRPATH.joinpath(model_name + ".yaml")
    assert config_filepath.exists(), f"config for model {model_name} does not exist"

    config = OmegaConf.load(config_filepath)

    assert (
        config.image_transform.image_size
        == config.model_cfg.params.vision_cfg.image_size
    ), "image transform size and model input image size should be equal"

    text_tokenizer = ClipTokenizer()

    transform = image_transform(
        image_size=config.image_transform.image_size,
        is_train=is_train,
        mean=config.image_transform.mean,
        std=config.image_transform.std,
        image_resize_res=None,
        interpolation_mode=InterpolationMode.BICUBIC,
    )

    model = build_model(
        model_name=config.model_cfg.name,
        **OmegaConf.to_container(config.model_cfg.params),
    )

    ckpt_path = _download_model_ckpt(
        config.checkpoint_url,
        download_root or os.path.expanduser("~/.cache/diht"),
    )
    model = _load_model_ckpt(model, ckpt_path)

    if is_train is False:
        model.eval()

    return text_tokenizer, transform, model
