import pickle

from pandas import read_csv
from inspect import getsourcefile
from os.path import abspath, dirname, basename, join


def file_path(local_object):
    return abspath(getsourcefile(local_object))


def file_dir(local_object):
    return dirname(file_path(local_object))


def file_name(local_object):
    return basename(file_path(local_object))


def read_file(*path, encoding='utf-8'):
    with open(join(*path), encoding=encoding) as file:
        return list(file.read().replace('\n', ''))


def make_paths(file_names, path='', ext=''):
    if ext:
        return [f'{path}/{name}.{ext}' for name in file_names]
    else:
        return [f'{path}/{name}' for name in file_names]


def pickle_obj(obj, file_path, ext='pickle'):
    with open(f'{file_path}.{ext}', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def unpickle_file(file_path, ext='pickle'):
    with open(f'{file_path}.{ext}', 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_csv(file_path,
               ext='pickle',
               nrows=None,
               usecols=None,
               index_col=None,
               dtype_dict=None,
               header='infer'):
    '''
    Use pandas to preserve dtypes!
    '''
    csv_data = read_csv(f'{file_path}.csv',
                        nrows=nrows,
                        usecols=usecols,
                        index_col=index_col,
                        dtype=dtype_dict,
                        header=header)
    pickle_obj(csv_data, file_path)