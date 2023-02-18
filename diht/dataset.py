# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
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


class ImageMultiCaptionDataset(Dataset):
    r"""Generic class for image-caption datasets where each image is associated with
        more than one caption.
    Methods:
    --------
    __getitem__(index): Tuple
        Return a tuple (image, captions). Both are preprocessed with
        respective transforms.
    len(): int
        Return the number of samples.
    Attributes:
    number_of_captions: int
        Return the number of captions in the dataset.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for ImageMultiCaptionDataset.
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        self._root = root
        self._name = name
        self._split = split
        self._transform = transform
        self._load_data()

    def _load_data(self) -> None:
        with open(os.path.join(self._root, f"{self._name}_{self._split}.json")) as f:
            annotations = json.load(f)
        processed_data = self._prepare_data_from_annotations(
            annotations, root=self._root
        )
        self._text = processed_data["text"]
        self._image = processed_data["image_ids"]
        self._txt2img = processed_data["txt2img"]
        self._img2txt = processed_data["img2txt"]
        self._data = processed_data["image_captions_data"]
        self._number_of_captions = int(sum(len(captions) for _, captions in self._data))

    def _prepare_data_from_annotations(self, annotations, root=None):
        text = []
        image_ids = []
        txt2img = {}
        img2txt = {}
        image_captions_data = []

        txt_id = 0
        for img_id, ann in enumerate(annotations):
            image_i = os.path.join(root, ann["image"])
            image_ids.append(image_i)
            img2txt[img_id] = []
            captions_i = []
            if isinstance(ann["caption"], str):
                ann_caption = [ann["caption"]]
            elif isinstance(ann["caption"], list):
                ann_caption = ann["caption"]
            else:
                raise TypeError("'str' or 'list' allowed for captions")

            for caption in ann_caption:
                text.append(caption)
                captions_i.append(caption)
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

            image_captions_data.append((image_i, captions_i))

        return {
            "text": text,
            "image_ids": image_ids,
            "txt2img": txt2img,
            "img2txt": img2txt,
            "image_captions_data": image_captions_data,
        }

    def __len__(self) -> int:
        r"""Return number of images in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        len: int
            Number of images in the dataset.
        """
        return len(self._data)

    @property
    def number_of_captions(self) -> int:
        r"""Return number of captions in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        number_of_captions: int
            Number of captions in the dataset.
        """
        return self._number_of_captions

    def __getitem__(self, index: int) -> Union[Tuple, None]:
        r"""Return sample as tuple (image, index).
        We only return image and index because there could be varying number of
        captions per image and it could break default collates.
        The class implements _load_text to handle loading captions per image.
        Parameters:
        -----------
        index: int
            The index of the sample.
        Returns:
        --------
        image: array_like
            The image for the `index` sample.
        index: int
            The index of the sample.
        Exception handling:
        -------------------
        If an exception is raised during dataloading,
        this function returns `None`.
        """
        image = self._load_image(index)
        return image, index

    def _load_image(self, index: int) -> Tuple:
        path, _ = self._data[index]
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image

    def _load_captions(self, index):
        _, captions = self._data[index]
        if isinstance(captions, str):
            captions = [captions]
        return captions

    def _load_text(self, index: int) -> Tuple:
        captions = self._load_captions(index)
        return captions, self._img2txt[index]

    def _load_item(self, index: int) -> Tuple:
        image = self._load_image(index)
        captions = self._load_captions(index)
        return image, captions


class COCO(ImageMultiCaptionDataset):
    r"""The COCO Captions dataset.
    Webpage: https://cocodataset.org/#download
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for COCO
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        super().__init__(
            root=root,
            name="coco",
            split=split,
            transform=transform,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "COCO(",
                f"  split={self._split},",
                f"  n_images={self.__len__()},",
                f"  n_captions={self.number_of_captions}"
                f"  transform={self._transform}",
                ")",
            ]
        )


class Flickr30K(ImageMultiCaptionDataset):
    r"""The Flickr30K Captions dataset.
    Webpage: https://shannon.cs.illinois.edu/DenotationGraph/
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for COCO
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        super().__init__(
            root=root,
            name="flickr30k",
            split=split,
            transform=transform,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Flickr30K(",
                f"  split={self._split},",
                f"  n_images={self.__len__()},",
                f"  n_captions={self.number_of_captions}"
                f"  transform={self._transform}",
                ")",
            ]
        )
