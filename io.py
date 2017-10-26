# Read and write
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


import pickle
from inspect import getsourcefile
from os.path import abspath, basename, dirname, join
from typing import Any, Iterable, List

from pandas import read_csv
from pyscripts.misc import strip_ext


def file_path(
        local_object: Any) \
        -> str:
    return abspath(getsourcefile(local_object))


def file_dir(
        local_object: Any) \
        -> str:
    return dirname(file_path(local_object))


def file_name(
        local_object: Any) \
        -> str:
    return basename(file_path(local_object))


def read_file(
        *path: str,
        encoding: str = 'utf-8') \
        -> List[str]:
    with open(join(*path), encoding = encoding) as file:
        return list(file.read().replace('\n', ''))


def pickle_obj(
        obj: Any,
        save_path: str,
        ext: str = 'pickle') \
        -> None:
    save_path = strip_ext(save_path, ext)
    with open(f'{save_path}.{ext}', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def unpickle_file(
        path: str,
        ext: str = 'pickle') \
        -> Any:
    path = strip_ext(path, ext)
    with open(f'{path}.{ext}', 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_csv(
        csv_path: str,
        save_path: str,
        ext: str = 'csv',
        nrows: int = None,
        usecols: Iterable[int] = None,
        index_col: int = None,
        dtype_dict: dict = None,
        header: str = 'infer') \
        -> None:
    # Use pandas to preserve dtypes!
    csv_path = strip_ext(csv_path, ext)
    csv_data = read_csv(f'{csv_path}.{ext}',
                        nrows = nrows,
                        usecols = usecols,
                        index_col = index_col,
                        dtype = dtype_dict,
                        header = header)
    pickle_obj(csv_data, save_path)
