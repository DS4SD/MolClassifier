import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops import box_area
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

print_time = False


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class Resize(nn.Module):
    def __init__(
        self,
        size: List[int],
        interpolation: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._resize = T.Resize(size, max_size=max_size)  # interpolation=interpolation,

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        t = -time.time()
        # Resize image
        shape = image.shape
        old_dims = torch.FloatTensor([shape[1], shape[2], shape[1], shape[2]]).unsqueeze(0)
        image = self._resize(image)

        if target is not None:
            if "masks" in target:
                target["masks"] = self._resize(target["masks"])
            #
            if "boxes" in target:
                new_boxes = target["boxes"] / old_dims
                shape = image.shape
                new_dims = torch.FloatTensor([shape[1], shape[2], shape[1], shape[2]]).unsqueeze(0)
                target["boxes"] = new_boxes * new_dims
                target["area"] = box_area(target["boxes"])
            #
        #
        t += time.time()
        if print_time:
            print("resize:", t)

        return image, target

    #


#


class GaussianBlur(nn.Module):
    def __init__(
        self,
        kernel_size: List[int],
        # sigma: Optional[Tuple[float]] = None
    ) -> None:
        super().__init__()
        self._gaussianblur = T.GaussianBlur(kernel_size)  # , sigma=sigma)

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        t = -time.time()
        # Resize image
        image = self._gaussianblur(image)

        if target is not None:
            if "masks" in target:
                target["masks"] = self._gaussianblur(target["masks"])
            #
            """
            if "boxes" in target:
                new_boxes = target["boxes"] / old_dims
                shape = image.shape
                new_dims = torch.FloatTensor([shape[1], shape[2], shape[1], shape[2]]).unsqueeze(0)
                target["boxes"] = new_boxes * new_dims
                target["area"] = box_area(target["boxes"])
            #
            """
        #
        t += time.time()
        if print_time:
            print("resize:", t)

        return image, target

    #


#
