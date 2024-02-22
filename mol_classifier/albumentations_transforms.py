import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class PixelSpotNoise(ImageOnlyTransform):
    """Apply pixel noise to the input image.
    Args:
        value ((float, float, float) or float): color value of the pixel.
        prob (float): probability to add pixels
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_holes=1,
        max_holes=8,
        min_height=1,
        max_height=8,
        min_width=1,
        max_width=8,
        prob=0.05,
        value=0,
        always_apply=False,
        p=0.5,
    ):
        super(PixelSpotNoise, self).__init__(always_apply, p)

        if isinstance(prob, (list, tuple)):
            if len(prob) != 2:
                raise ValueError("prob should be a minx/max pair.")
            if 0 < prob[0] > 1:
                raise ValueError("prob should be in range [0,1].")
            if 0 < prob[1] > 1:
                raise ValueError("prob should be in range [0,1].")
            if prob[1] < prob[0]:
                raise ValueError(f"prob should be [min, max], got {prob}.")
        elif isinstance(prob, (int, float)):
            if prob < 0:
                raise ValueError("prob should be non negative.")
            if prob > 1:
                raise ValueError("prob should be smaller than 1.")
            prob = (prob,) * 2
        else:
            raise TypeError("Expected prob type to be one of (int, float), got {}".format(type(prob)))

        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError("value should be non negative.")
        else:
            raise TypeError("Expected value type to be one of (int, float, tuple, list), got {}".format(type(value)))

        self.prob = prob
        self.value = value

        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

    def apply(self, img, **params):
        prob = (self.prob[1] - self.prob[0]) * np.random.random_sample() + self.prob[0]
        holes = self._get_hole_list(img.shape[:2])
        for x1, y1, x2, y2 in holes:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if np.random.random_sample() <= prob:
                        img[y, x] = self.value
                    #
                #
            #
        #
        return img

    def get_transform_init_args(self):
        return {"value": self.value, "prob": self.prob}

    def _get_hole_list(self, img_shape):
        height, width = img_shape[:2]

        holes = []
        for _n in range(np.random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                ]
            ):
                hole_height = np.random.randint(self.min_height, self.max_height)
                hole_width = np.random.randint(self.min_width, self.max_width)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                ]
            ):
                hole_height = int(height * np.random.uniform(self.min_height, self.max_height))
                hole_width = int(width * np.random.uniform(self.min_width, self.max_width))
            else:
                raise ValueError(
                    "Min width, max width,                     min height and max"
                    " height                     should all either be ints or floats.  "
                    "                   Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                        ]
                    )
                )
            #
            y1 = max(0, np.random.randint(-hole_height + 1, height + hole_height - 1))
            x1 = max(0, np.random.randint(-hole_width + 1, width + hole_width - 1))
            y2 = min(y1 + hole_height, height)
            x2 = min(x1 + hole_width, width)
            holes.append((x1, y1, x2, y2))
        #
        return holes


class PixelNoise(ImageOnlyTransform):
    """Apply pixel noise to the input image.
    Args:
        value ((float, float, float) or float): color value of the pixel.
        prob (float): probability to add pixels
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, prob=0.05, value=0, always_apply=False, p=0.5):
        super(PixelNoise, self).__init__(always_apply, p)

        if isinstance(prob, (list, tuple)):
            if len(prob) != 2:
                raise ValueError("prob should be a minx/max pair.")
            if 0 < prob[0] > 1:
                raise ValueError("prob should be in range [0,1].")
            if 0 < prob[1] > 1:
                raise ValueError("prob should be in range [0,1].")
            if prob[1] < prob[0]:
                raise ValueError(f"prob should be [min, max], got {prob}.")
        elif isinstance(prob, (int, float)):
            if prob < 0:
                raise ValueError("prob should be non negative.")
            if prob > 1:
                raise ValueError("prob should be smaller than 1.")
            prob = (prob,) * 2
        else:
            raise TypeError("Expected prob type to be one of (int, float), got {}".format(type(prob)))

        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError("value should be non negative.")
        else:
            raise TypeError("Expected value type to be one of (int, float, tuple, list), got {}".format(type(value)))

        self.prob = prob
        self.value = value

    def apply(self, img, **params):
        prob = (self.prob[1] - self.prob[0]) * np.random.random_sample() + self.prob[0]
        mask = np.random.choice([True, False], size=img.shape, p=[prob, 1.0 - prob])
        if len(img.shape) == 2:
            img[mask] = self.value
        elif len(img.shape) == 3:
            for i in range(img.shape[0]):
                img[i, mask] = self.value
        else:
            raise ValueError(f"wrong image shape (2 or 3 valid), got {len(img.shape)}.")

        return img

    def get_transform_init_args(self):
        return {"value": self.value, "prob": self.prob}
