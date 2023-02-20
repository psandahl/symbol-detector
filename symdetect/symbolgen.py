import random

from PIL import Image, ImageDraw


def random_hud(size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    hud = Image.new('L', size)

    fill = random.randint(0, 255)
    draw = ImageDraw.Draw(hud)

    # Hair cross.
    hair_cross(draw, fill, size)

    # Create binary mask image from the hud.
    mask = hud.point(lambda i: 255 if i > 0 else 0).convert('1')

    return hud, mask


def hair_cross(draw: ImageDraw, fill: int, size: tuple[int, int]) -> None:
    useit = random.randint(0, 10)

    if useit < 10:
        if useit % 2 == 0:
            fill = random.randint(0, 255)

        thickness = random.randint(1, 4)

        width, height = size
        midx = (width - 1) // 2
        midy = (height - 1) // 2
        radius = random.randint(int(width * 0.02), int(width * 0.05))

        draw.line([(midx - radius, midy - radius), (midx + radius,
                  midy + radius)], width=thickness, fill=fill)
        draw.line([(midx - radius, midy + radius), (midx + radius,
                  midy - radius)], width=thickness, fill=fill)
