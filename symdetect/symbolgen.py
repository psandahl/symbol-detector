from __future__ import annotations

import pathlib
import random

from PIL import Image, ImageDraw
from tensorflow import keras

# class FilesSequence


def from_image_path(path: pathlib.Path) -> tuple[Image.Image, Image.Image]:
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
    print(f'prob={prob}')

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
    mask = hud.point(lambda i: 255 if i > 0 else 0).convert('1')

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
