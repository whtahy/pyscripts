import pickle
import string

from pandas import read_csv
from inspect import getsourcefile
from os.path import abspath, dirname, basename, join

from .misc import strip_ext


def file_path(local_object):
    return abspath(getsourcefile(local_object))


def file_dir(local_object):
    return dirname(file_path(local_object))


def file_name(local_object):
    return basename(file_path(local_object))


def read_file(*path, encoding='utf-8'):
    with open(join(*path), encoding=encoding) as file:
        return list(file.read().replace('\n', ''))


def pickle_obj(obj, save_path, ext='pickle'):
    save_path = strip_ext(save_path, ext)
    with open(f'{save_path}.{ext}', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def unpickle_file(file_path, ext='pickle'):
    file_path = strip_ext(file_path, ext)
    with open(f'{file_path}.{ext}', 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_csv(csv_path,
               save_path,
               ext='pickle',
               nrows=None,
               usecols=None,
               index_col=None,
               dtype_dict=None,
               header='infer'):
    '''
    Use pandas to preserve dtypes!
    '''
    csv_path = strip_ext(csv_path, 'csv')
    csv_data = read_csv(f'{csv_path}.csv',
                        nrows=nrows,
                        usecols=usecols,
                        index_col=index_col,
                        dtype=dtype_dict,
                        header=header)
    pickle_obj(csv_data, save_path)