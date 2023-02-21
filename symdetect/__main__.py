import argparse
import pathlib
import random

import symdetect.imagedataset as id
from symdetect.symbolgen import FilesSequence
import symdetect.unet as unet

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array

image_size = (256, 256)
batch_size = 5


def train_gryphon(dir: pathlib.Path, model_path: pathlib.Path) -> None:
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

    if model_path is not None:
        # Train model.
        unet.build_and_train(train_seq=train_seq, validation_seq=validation_seq,
                             image_size=image_size, model_path=model_path)
    else:
        # Visualize a few images from a batch.
        X, Y = train_seq[random.randint(0, len(train_seq))]

        plt.figure(figsize=(9, 9))
        for num, idx in enumerate([0, 1, 2, 3]):
            plt.subplot(4, 2, 2 * num + 1)
            plt.imshow(X[idx], cmap='gray')
            plt.title('image with HUD')
            plt.axis('off')

            plt.subplot(4, 2, 2 * num + 2)
            plt.imshow(Y[idx], cmap='gray')
            plt.title('mask')
            plt.axis('off')

        plt.show()


def predict_mask(image_path: pathlib.Path, model_path: pathlib.Path) -> None:
    im = keras.utils.load_img(image_path, color_mode='grayscale',
                              target_size=image_size, interpolation='bicubic')
    im = img_to_array(im) / 255.

    model = keras.models.load_model(model_path)

    plt.figure(figsize=(9, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')

    im = np.expand_dims(im, 0)
    result = model.predict(im)

    plt.subplot(1, 2, 2)
    plt.imshow(result[0], cmap='gray')
    plt.axis('off')

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gryphon', type=pathlib.Path,
                        help='Train using Gryphon dataset')
    parser.add_argument('--model', type=pathlib.Path,
                        help='Path to a model file to read or write')
    parser.add_argument('--predict', type=pathlib.Path,
                        help='Path to image for which a mask will be predicted')
    opts = parser.parse_args()

    if opts.gryphon is not None:
        train_gryphon(opts.gryphon, opts.model)
    elif opts.predict is not None and opts.model is not None:
        predict_mask(opts.predict, opts.model)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
