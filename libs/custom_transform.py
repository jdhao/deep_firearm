"""
this script implement some custom transform functions
"""

from PIL import Image, ImageOps, ImageEnhance
import random
from torchvision.transforms import Compose, Lambda
import numpy as np

__all__ = ["RandomResize",  "RandomRotate", "ColorJitter", "Resize"]


class RandomResize(object):
    """
    Resize the longest side of an image to a random size in the given range
    """
    def __init__(self, min_size, max_size, step=8,
                 interpolation=Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.step = step

    def __call__(self, img):
        size_bound = random.randrange(self.min_size, self.max_size+1, self.step)

        old_size = (img.width, img.height)
        ratio = float(size_bound)/max(old_size)
        new_size = [int(x*ratio) for x in old_size]

        return img.resize(new_size, self.interpolation)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(self.size)/max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        return img.resize(new_size, resample=self.interpolation)


class ResizePad(object):
    """
    Resize max length of input image to given size, and padd image in shorter
    side to form a square image
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(self.size)/max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        delta_w = self.size - new_size[0]
        delta_h = self.size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2),
                   delta_h - (delta_h // 2))
        new_img = ImageOps.expand(img, padding, fill="black")

        return new_img


class RandomRotate(object):
    """
    randomly rotate the image by a degree in range [-angle, angle]
    """
    def __init__(self, angle, interpolation=Image.BILINEAR):
        self.angle = angle
        self.interpolation = interpolation

    def __call__(self, img):
        degree = random.uniform(-self.angle, self.angle)

        return img.rotate(degree, resample=self.interpolation)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    @staticmethod
    def get_params(brightness, contrast, saturation):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness),
                                                  1 + brightness)
            transforms.append(Lambda(
                lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast),
                                                1 + contrast)
            transforms.append(
                Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation),
                                                  1 + saturation)
            transforms.append(Lambda(
                lambda img: adjust_saturation(img, saturation_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation)

        return transform(img)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img