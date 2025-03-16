import random

import numpy as np
from torchvision import transforms


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_img, post_img):  # , label):
        for t in self.transforms:
            sample = t(pre_img, post_img)  # , label)
            # pre_img, post_img, label = (
            pre_img, post_img = (
                sample["pre_image"],
                sample["post_image"],
                # sample["label"],
            )

        return {"pre_image": pre_img, "post_image": post_img}  # , "label": label}


# class RandomCrop:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, pre_img, post_img, label):
#         h, w = pre_img.shape[:2]
#         new_h, new_w = self.size, self.size

#         if h > new_h and w > new_w:
#             top = random.randint(0, h - new_h)
#             left = random.randint(0, w - new_w)
#         else:
#             # Pad if image is smaller than crop size
#             pad_h = max(0, new_h - h)
#             pad_w = max(0, new_w - w)

#             pre_img = np.pad(pre_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
#             post_img = np.pad(
#                 post_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
#             )
#             label = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

#             top, left = 0, 0

#         pre_img = pre_img[top : top + new_h, left : left + new_w]
#         post_img = post_img[top : top + new_h, left : left + new_w]
#         label = label[top : top + new_h, left : left + new_w]

#         return {"pre_image": pre_img, "post_image": post_img, "label": label}


class RandomFlip:
    def __call__(self, pre_img, post_img):  # , label):
        if random.random() < 0.5:
            pre_img = np.fliplr(pre_img).copy()
            post_img = np.fliplr(post_img).copy()
            # label = np.fliplr(label).copy()

        if random.random() < 0.5:
            pre_img = np.flipud(pre_img).copy()
            post_img = np.flipud(post_img).copy()
            # label = np.flipud(label).copy()

        return {"pre_image": pre_img, "post_image": post_img}  # , "label": label}


class RandomRotation:
    def __call__(self, pre_img, post_img):  # , label):
        k = random.randint(0, 3)
        pre_img = np.rot90(pre_img, k=k).copy()
        post_img = np.rot90(post_img, k=k).copy()
        # label = np.rot90(label, k=k).copy()

        return {"pre_image": pre_img, "post_image": post_img}  # , "label": label}


def get_transform(split, crop_size=256):
    if split == "train":
        return Compose(
            [
                # RandomCrop(crop_size),
                RandomFlip(),
                RandomRotation(),
            ]
        )
    else:
        return None
