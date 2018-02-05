# Read and write
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import os
import pickle
from inspect import getsourcefile
from os.path import abspath, basename, dirname
from typing import TYPE_CHECKING

import numpy
from pandas import read_csv

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def file_path(
        obj: 'Any') \
        -> str:
    return abspath(getsourcefile(obj))


def file_dir(
        obj: 'Any') \
        -> str:
    return dirname(file_path(obj))


def file_name(
        obj: 'Any') \
        -> str:
    return basename(file_path(obj))


def read_file(
        path: str,
        encoding: str = 'utf-8',
        line_filter: 'Callable[str, bool]' = lambda x: True,
        break_condition: 'Callable[str, bool]' = lambda x: False) \
        -> 'List[str]':
    out = []
    with open(path, encoding = encoding) as file:
        for line in file:
            if break_condition(line):
                break
            if line_filter(line):
                out += [line.rstrip('\n')]
    return out


def scan_dir(
        path: str,
        folder_filter: 'Callable[str, bool]' = lambda x: True,
        file_filter: 'Callable[str, bool]' = lambda x: True,
        recursive: bool = True) \
        -> 'List[str]':
    out = []
    for dir_name, subdir_names, file_names in os.walk(path):
        if recursive:
            subdir_names[:] = [name for name in subdir_names if
                               folder_filter(name)]
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
        -> 'ndarray':
    with open(path, encoding = encoding) as file:
        return numpy.fromiter(file.read().replace('\n', ''),
                              dtype = f'U{chunk_size}')


def pickle_obj(
        obj: 'Any',
        save_path: str) \
        -> None:
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def pickle_csv(
        csv_path: str,
        save_path: str,
        nrows: int = None,
        usecols: 'NLT_IntType' = None,
        index_col: int = None,
        dtype_dict: dict = None,
        header: str = 'infer') \
        -> None:
    # Use pandas to preserve dtypes!
    csv_data = read_csv(csv_path,
                        nrows = nrows,
                        usecols = usecols,
                        index_col = index_col,
                        dtype = dtype_dict,
                        header = header)
    pickle_obj(csv_data, save_path)


def unpickle_file(
        path: str) \
        -> 'Any':
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_NLT(
        arr: 'NLT_Type',
        save_path: 'str') \
        -> None:
    with open(save_path, 'w') as file:
        for line in arr:
            file.write(str(line) + '\n')
