import random

import numpy as np
from torchvision import transforms


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_img, post_img):
        for t in self.transforms:
            sample = t(pre_img, post_img)
            pre_img, post_img = (
                sample["pre_image"],
                sample["post_image"],
            )

        return {"pre_image": pre_img, "post_image": post_img}


class RandomFlip:
    def __call__(self, pre_img, post_img):
        if random.random() < 0.5:
            pre_img = np.fliplr(pre_img).copy()
            post_img = np.fliplr(post_img).copy()

        if random.random() < 0.5:
            pre_img = np.flipud(pre_img).copy()
            post_img = np.flipud(post_img).copy()

        return {"pre_image": pre_img, "post_image": post_img}


class RandomRotation:
    def __call__(self, pre_img, post_img):
        k = random.randint(0, 3)
        pre_img = np.rot90(pre_img, k=k).copy()
        post_img = np.rot90(post_img, k=k).copy()

        return {"pre_image": pre_img, "post_image": post_img}


def get_transform(split, crop_size=256):
    if split == "train":
        return Compose(
            [
                RandomFlip(),
                RandomRotation(),
            ]
        )
    else:
        return None
