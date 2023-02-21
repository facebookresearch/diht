# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from diht import model_zoo
from diht.dataset import ImageNet1K
from diht.dataset_helper import (
    ZEROSHOT_IMAGENET1K_CLASSNAMES,
    ZEROSHOT_IMAGENET1K_TEMPLATES,
)

from torch.utils.data import DataLoader
from tqdm import tqdm


IMAGENET_ROOT = (
    "<YOUR_IMAGENET_ROOT_HERE>"  # replace with your ImageNet1K root directory
)
MODEL_NAME = "diht_vitl14_336px"  # replace with the model you want to evaluate


def accuracy(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())


def main():
    # get tokenizer, transform, model, eval dataset and dataloader
    print(f"Load model {MODEL_NAME} ...")
    text_tokenizer, image_transform, model = model_zoo.load_model(
        MODEL_NAME, is_train=False
    )
    eval_dataset = ImageNet1K(
        root=IMAGENET_ROOT, split="val", transform=image_transform
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=32,
        num_workers=5,
    )

    # move model to GPU
    model = model.cuda()

    # init accuracy counters
    acc_1 = 0.0
    num_samples = 0.0

    # compute classifiers for zero-shot classification
    print("Extract zero-shot classifiers for ImageNet1K ...")
    with torch.no_grad():
        zeroshot_classifiers = []
        for classname in tqdm(ZEROSHOT_IMAGENET1K_CLASSNAMES):
            texts = [
                template.format(classname) for template in ZEROSHOT_IMAGENET1K_TEMPLATES
            ]  # format with class
            texts = text_tokenizer(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_classifiers.append(class_embedding)
        zeroshot_classifiers = torch.stack(zeroshot_classifiers, dim=1).cuda()

    # compute image features
    print("Extract image features for ImageNet1K ...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            images, labels = batch[0].cuda(), batch[1].cuda()
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ zeroshot_classifiers
            acc_1 += accuracy(logits, labels)
            num_samples += images.size(0)
    print(f"ImageNet1K acc@1 for {MODEL_NAME}: {100*acc_1/num_samples:.1f}")


if __name__ == "__main__":
    main()
