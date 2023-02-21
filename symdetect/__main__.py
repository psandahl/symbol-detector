import argparse
import pathlib
import random

import symdetect.imagedataset as id
import symdetect.symbolgen as sg

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


def train_gryphon(dir: pathlib.Path) -> None:
    files = id.gryphon_image_paths(dir)

    if files is None:
        return None

    # Shuffle images.
    random.Random(8080).shuffle(files)

    # Divide into train and test image sets.
    num_test_files = int(len(files) * 0.1)
    train_set = files[:-num_test_files]
    test_set = files[-num_test_files:]

    print(f'Total number of JPG images found={len(files)}')
    print(f' Train files={len(train_set)}')
    print(f' Test files={len(test_set)}')

    random.seed(177)

    plt.figure(figsize=(9, 9))

    for num, idx in enumerate([12, 98, 456]):
        im, mask = sg.from_image_path(train_set[idx])
        im = img_to_array(im).astype(np.float32) / 255.
        mask = img_to_array(mask).astype(np.float32)

        plt.subplot(3, 2, 2 * num + 1)
        plt.imshow(im, cmap='gray')
        plt.subplot(3, 2, 2 * num + 2)
        plt.imshow(mask, cmap='gray')

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gryphon', type=pathlib.Path,
                        help='Train using Gryphon dataset')
    opts = parser.parse_args()

    if opts.gryphon is not None:
        train_gryphon(opts.gryphon)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
