from __future__ import annotations

import pathlib
import random

import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array


class FilesSequence(keras.utils.Sequence):
    """
    Sequence for image based training datasets.
    """

    def __init__(self: FilesSequence, image_set: list[pathlib.Path],
                 image_size: tuple[int, int], batch_size: int,
                 seed: int = 1598) -> None:
        """
        Create the FilesSequence.

        Parameters:
            image_set: List of image paths.
            image_size: The image size to resize images to.
            batch_size: The batch size, that each call to __getitem__ shall give.
            seed: Random seed.
        """
        self.image_set = image_set
        self.image_size = image_size
        self.batch_size = batch_size

        random.seed(seed)

    def __len__(self: FilesSequence) -> int:
        return len(self.image_set) // self.batch_size

    def __getitem__(self: FilesSequence, idx: int) -> tuple[np.array, np.array]:
        X = np.zeros((self.batch_size,) + self.image_size +
                     (1,), dtype=np.float32)
        Y = np.zeros_like(X)

        base = idx * self.batch_size
        for i, j in enumerate(range(base, base + self.batch_size)):
            # Get image + mask.
            im, mask = from_image_path(self.image_set[j])

            # Resize image and mask
            im = im.resize(self.image_size[::-1])
            mask = mask.resize(self.image_size[::-1])

            # Convert to float32.
            im = img_to_array(im).astype(np.float32) / 255.
            mask = img_to_array(mask).astype(np.float32) / 255.

            # Copy to batch.
            X[i] = im
            Y[i] = mask

        return X, Y


def from_image_path(path: pathlib.Path) -> tuple[Image.Image, Image.Image]:
    """
    Read an image from the path. Annotate it with a HUD, and return the
    image with a mask.
    """
    with Image.open(path, mode='r') as im:
        im = im.convert(mode='L')
        hud, mask = random_hud(im.size)
        im = Image.composite(hud, im, mask)

        return im, mask


def random_hud(size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    """
    Generate a random hud with both reasonable and unreasonable items.
    """
    hud = Image.new('L', size, color=0)

    # Generate one from the "lottery".
    prob = random.randint(1, 100)

    no_hud = 25
    if prob % no_hud != 0:
        width, height = size
        cx = (width - 1) // 2
        cy = (height - 1) // 2

        # Draw object.
        draw = ImageDraw.Draw(hud)
        draw.fontmode = 'L'

        # Base fill color. Can be overridden.
        fill = random.randint(0, 255)

        # Hair cross in center.
        if prob % 2 == 0:
            if prob % 10 == 0:
                cross_fill = random.randint(0, 255)
            else:
                cross_fill = fill
            cross_at(draw,
                     position=(cx, cy),
                     radius=random.randint(3, 8),
                     width=random.randint(1, 4),
                     fill=cross_fill)

        # Center box.
        if prob % 4 == 0 or prob % 11 == 0:
            offs = random.randint(int(width * 0.02), int(width * 0.1))
            radius = random.randint(3, 8)
            line_width = random.randint(1, 4)
            hook_at(draw, type=0,
                    position=(cx - offs, cy - offs),
                    radius=radius,
                    width=line_width,
                    fill=fill)
            hook_at(draw, type=1,
                    position=(cx - offs, cy + offs),
                    radius=radius,
                    width=line_width,
                    fill=fill)
            hook_at(draw, type=2,
                    position=(cx + offs, cy + offs),
                    radius=radius,
                    width=line_width,
                    fill=fill)
            hook_at(draw, type=3,
                    position=(cx + offs, cy - offs),
                    radius=radius,
                    width=line_width,
                    fill=fill)

    # Create binary mask image from the hud.
    mask = hud.point(lambda i: 255 if i > 0 else 0).convert('L')

    return hud, mask


def hook_at(draw: ImageDraw, type: int, position: tuple[int, int],
            radius: int, width: int, fill: int) -> None:
    assert type >= 0 and type < 4
    cx, cy = position

    p0 = position
    if type == 0:  # upper left
        p1 = (cx + radius, cy)
        p2 = (cx, cy + radius)
    elif type == 1:  # lower left
        p1 = (cx, cy - radius)
        p2 = (cx + radius, cy)
    elif type == 2:  # lower right
        p1 = (cx, cy - radius)
        p2 = (cx - radius, cy)
    elif type == 3:  # upper right
        p1 = (cx, cy + radius)
        p2 = (cx - radius, cy)

    draw.line([p0, p1, p0, p2], width=width, fill=fill)


def cross_at(draw: ImageDraw, position: tuple[int, int], radius: int, width: int, fill: int) -> None:
    cx, cy = position

    draw.line([(cx - radius, cy - radius),
              (cx + radius, cy + radius),], width=width, fill=fill)
    draw.line([(cx - radius, cy + radius),
              (cx + radius, cy - radius)], width=width, fill=fill)
