# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def image_transform(
    image_size,
    is_train,
    mean,
    std,
    image_resize_res,
    interpolation_mode,
):
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that
        # Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if image_resize_res is None:
        image_resize_res = image_size
    else:
        if (
            isinstance(image_resize_res, (list, tuple))
            and image_resize_res[0] == image_resize_res[1]
        ):
            image_resize_res = image_resize_res[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose(
            [
                RandomResizedCrop(
                    image_size,
                    scale=(0.9, 1.0),
                    interpolation=interpolation_mode,
                ),
                ToTensor(),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                Resize(image_resize_res, interpolation=interpolation_mode),
                CenterCrop(image_size),
                ToTensor(),
                normalize,
            ]
        )
