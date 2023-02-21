from __future__ import annotations

import pathlib

from tensorflow import keras


class FileSequence(keras.utils.Sequence):
    def __init__(self: FileSequence):
        pass


def gryphon_image_paths(dir: pathlib.Path) -> list[pathlib.Path]:
    # /home/patsa/datasets/gryphon/Gryphon_dataset/Gryphon_dataset
    if not dir.exists() or not dir.is_dir():
        print(f"Directory '{dir}' must exist")
        return None

    # Expect to find 'Raw_images' under the given directory.
    dir = dir / 'Raw_images'
    if not dir.exists() or not dir.is_dir():
        print("Subdirectory 'Raw_images' must exist")
        return None

    files = list()
    __recursively_find_jpg(dir, files)

    return files


def __recursively_find_jpg(dir: pathlib.Path, files: list[pathlib.Path]) -> None:
    assert dir.exists() and dir.is_dir()

    for child in dir.iterdir():
        if child.is_dir():
            __recursively_find_jpg(child, files)
        elif child.is_file() and child.suffix == '.jpg':
            files.append(child)
