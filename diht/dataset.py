# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Callable, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class ImageNet1K(Dataset):
    r"""The ImageNet-1K dataset.

    The ImageNet-1K dataset spans 1000 object classes and contains
    1,281,671 training images, 50,000 validation images and 100,000
    test images.

    Webpage: https://www.image-net.org/
    Reference: https://ieeexplore.ieee.org/abstract/document/5206848
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for ImageNet1K

        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val).
        transform: Callable
            An image transformation.

        Returns:
        --------
        None
        """
        super().__init__()
        self._root = root
        self._split = split
        self._transform = transform
        self._data = self._load_data()

    def _load_data(self) -> None:
        directory = os.path.join(
            self._root,
            self._split,
        )
        _, synset_to_idx = find_classes(directory)
        data = make_dataset(
            directory=directory,
            class_to_idx=synset_to_idx,
            extensions=IMG_EXTENSIONS,
        )
        return data

    def __len__(self) -> int:
        r"""Return number of samples in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        len: int
            Number of samples in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Union[Tuple, None]:
        r"""Return sample as tuple (image, caption).
        Parameters:
        -----------
        index: int
            The index of the sample.
        Returns:
        --------
        image: array_like
            The image for the `index` sample.
        label: int
            The class label for the `index` sample.
        """
        path, label = self._data[index]
        image = Image.open(path).convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        return image, label

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ImageNet1K(",
                f"  split={self._split},",
                f"  n_samples={self.__len__()},",
                f"  transform={self._transform}",
                ")",
            ]
        )
