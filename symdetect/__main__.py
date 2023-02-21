import argparse
import pathlib
import random

import symdetect.imagedataset as id
from symdetect.symbolgen import FilesSequence
import symdetect.unet as unet

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


def train_gryphon(dir: pathlib.Path) -> None:
    image_size = (256, 256)
    batch_size = 5

    # Load files from the dataset.
    files = id.gryphon_image_paths(dir)

    if files is None:
        return None

    # Shuffle images.
    random.Random(8080).shuffle(files)

    # Divide into train and test image sets.
    num_test_files = int(len(files) * 0.1)
    train_set = files[:-num_test_files]
    validation_set = files[-num_test_files:]

    # Create test and validation sequences.
    train_seq = FilesSequence(
        train_set, image_size=image_size, batch_size=batch_size, seed=1598)
    validation_seq = FilesSequence(
        validation_set, image_size=image_size, batch_size=batch_size, seed=1033)

    print(f'Total number of JPG images found={len(files)}')
    print(f' Train files={len(train_set)}, train batches={len(train_seq)}')
    print(
        f' Validation files={len(validation_set)}, validation batches={len(validation_seq)}')

    unet.build_and_train(train_seq=train_seq, validation_seq=validation_seq,
                         image_size=image_size, model_path=pathlib.Path('models/firsttry.h5'))

    # Visualize a few images from a batch.
    # X, Y = train_seq[random.randint(0, len(train_seq))]

    # plt.figure(figsize=(9, 9))
    # for num, idx in enumerate([0, 3, 6, 14]):
    #    plt.subplot(4, 2, 2 * num + 1)
    #    plt.imshow(X[idx], cmap='gray')
    #    plt.title('image with HUD')
    #    plt.axis('off')

    #    plt.subplot(4, 2, 2 * num + 2)
    #    plt.imshow(Y[idx], cmap='gray')
    #    plt.title('mask')
    #    plt.axis('off')

    # plt.show()


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
