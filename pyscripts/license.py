# License utils
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import os
from functools import partial

import numpy
import pandas
from stdlib_list import stdlib_list

from pyscripts.hero import re_imports
from pyscripts.knots import str_replace, os_path
from pyscripts.scholar import dir_apply, file_apply, write_file

PROJECT_NAME = 'pyscripts'
PROJECT_ROOT = os_path(f'c:/git/' + PROJECT_NAME)

LICENSE_DIR = 'licenses'
LICENSE_PATH = PROJECT_ROOT + os.sep + LICENSE_DIR

EXCLUDE_DIRS = ['.idea', '.git', '__pycache__']
EXCLUDE_FILES = []
EXCLUDE_LIBS = [PROJECT_NAME]

LIB_ALIASES = {
    'sklearn': 'scikit-learn'
}


def g():
    def folder_filter(x):
        return x not in EXCLUDE_DIRS

    def file_filter(x):
        return x.endswith('.py') and x not in EXCLUDE_FILES

    def func_filter(x):
        return x is not None and x != ''

    def break_cond(x):
        return '=' in x or 'def' in x

    def line_func(x):
        re_match = re_imports.search(x)
        if re_match:
            import_string = re_match.group()
            remove_these = ['from', 'import'] + EXCLUDE_LIBS + [' ']
            lib_name = str_replace(import_string, remove_these)
            return LIB_ALIASES.get(lib_name, lib_name)
        else:
            return None

    file_func = partial(file_apply,
                        func = line_func,
                        output_filter = func_filter,
                        break_condition = break_cond)
    lib_names = dir_apply(PROJECT_ROOT,
                          folder_filter = folder_filter,
                          file_filter = file_filter,
                          func = file_func)
    lib_names = pandas.unique(lib_names)
    extlib_names = numpy.setdiff1d(lib_names, stdlib_list())
    stdlib_names = numpy.intersect1d(lib_names, stdlib_list())
    return extlib_names, stdlib_names


def f():
    extlib_names, stdlib_names = g()

    for n in extlib_names:
        print(n)

    for n in stdlib_names:
        print(n)

    write_file(extlib_names, 'requirements.txt')

    for name in extlib_names:
        os.makedirs(os.path.join(LICENSE_PATH, name), exist_ok = True)


# TODO: interactive browse GitHub with SVN export
# TODO: use weave inside some scholar function to join directories
if __name__ == '__main__':
    f()
