# Read and write
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt

import os
import pickle
from typing import Any, Callable, Sequence

import numpy as np


def read_file(
        path: str,
        encoding: str = 'utf-8',
        line_filter: Callable[str, bool] = lambda x: True,
        break_condition: Callable[str, bool] = lambda x: False) \
        -> Sequence[str]:
    """
    Read text into list.
    """
    out = []
    with open(path, encoding=encoding) as file:
        for line in file:
            if break_condition(line):
                break
            if line_filter(line):
                out += [line.rstrip('\n')]
    return out


def scan_dir(
        path: str,
        folder_filter: Callable[str, bool] = lambda x: True,
        file_filter: Callable[str, bool] = lambda x: True,
        recursive: bool = True) \
        -> Sequence[str]:
    """
    File index.
    """
    out = []
    for dir_name, subdir_names, file_names in os.walk(path):
        if recursive:
            subdir_names[:] = [
                name for name in subdir_names if folder_filter(name)
            ]
        else:
            subdir_names[:] = []

        for f in file_names:
            if file_filter(f):
                f_path = os.path.join(dir_name, f)
                out += [f_path]
    return out


def split_file(
        path: str,
        encoding: str = 'utf-8',
        chunk_size: int = 1) \
        -> np.ndarray:
    """
    Read text into `ndarray`, in chunks.
    """
    with open(path, encoding=encoding) as file:
        return np.fromiter(
            file.read().replace('\n', ''), dtype=f'U{chunk_size}')


def write(
        obj: Any,
        save_path: str) \
        -> None:
    """
    Pickle `obj`.
    """
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def read(
        path: str) \
        -> Any:
    """
    Unpickle file at `path`.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_arr():
    """
    Write each row of array as line of text.
    """
    pass
