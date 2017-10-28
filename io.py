# Read and write
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


import pickle
from inspect import getsourcefile
from os.path import abspath, basename, dirname, join
from typing import TYPE_CHECKING

from pandas import read_csv

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


# TODO: filter_file
# TODO: apply_file
# TODO: read_dir (recursive option)


def file_path(
        local_object: 'Any') \
        -> str:
    return abspath(getsourcefile(local_object))


def file_dir(
        local_object: 'Any') \
        -> str:
    return dirname(file_path(local_object))


def file_name(
        local_object: 'Any') \
        -> str:
    return basename(file_path(local_object))


def read_file(
        *path: str,
        encoding: str = 'utf-8') \
        -> 'List[str]':
    with open(join(*path), encoding = encoding) as file:
        return list(file.read().replace('\n', ''))


def pickle_obj(
        obj: 'Any',
        save_path: str) \
        -> None:
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def unpickle_file(
        path: str) \
        -> 'Any':
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_csv(
        csv_path: str,
        save_path: str,
        nrows: int = None,
        usecols: 'Union[Iterable[int] Iterable[str]]' = None,
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
