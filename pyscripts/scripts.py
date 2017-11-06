# User scripts
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import subprocess
from functools import partial
from typing import TYPE_CHECKING

import numpy
import pandas
import requests
from stdlib_list import stdlib_list

from pyscripts.hero import re_imports, re_lastslash, re_startword
from pyscripts.knots import str_replace
from pyscripts.scholar import dir_apply, file_apply, write_NLT

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def google(
        query: str) \
        -> str:
    rq = requests.get(f'https://www.google.com/search?q={query}')
    return rq.text


def cmd_run(command):
    # https://stackoverflow.com/a/4760517
    raw = subprocess.run(command, stdout = subprocess.PIPE).stdout
    return raw.decode('utf-8')


def scan_imports(
        project_root,
        project_name: 'str' = None,
        exclude_dirs: 'NLT_StrType' = None,
        exclude_files: 'NLT_StrType' = None,
        exclude_libs: 'NLT_StrType' = None,
        lib_aliases: dict = None,
        write_requirements: bool = True) \
        -> 'Tuple[ndarray, ndarray]':
    if project_name is None:
        project_name = re_lastslash.search(project_root).group(1)
    if exclude_dirs is None:
        exclude_dirs = ['.idea', '.git', '__pycache__']
    if exclude_files is None:
        exclude_files = []
    if exclude_libs is None:
        exclude_libs = [project_name]
    if lib_aliases is None:
        lib_aliases = {
            'sklearn': 'scikit-learn'
        }

    def folder_filter(x):
        return x not in exclude_dirs

    def file_filter(x):
        return x.endswith('.py') and x not in exclude_files

    def func_filter(x):
        return x is not None and x != ''

    def break_cond(x):
        return '=' in x or 'def' in x

    def line_func(x):
        re_match = re_imports.search(x)
        if re_match:
            import_string = re_match.group()
            remove_these = ['from', 'import'] + exclude_libs + [' ']
            lib_name = str_replace(import_string, remove_these)
            return lib_aliases.get(lib_name, lib_name)
        else:
            return None

    file_func = partial(file_apply,
                        func = line_func,
                        output_filter = func_filter,
                        break_condition = break_cond)
    lib_names = dir_apply(project_root,
                          folder_filter = folder_filter,
                          file_filter = file_filter,
                          func = file_func)
    lib_names = pandas.unique(lib_names)
    extlib_names = numpy.setdiff1d(lib_names, stdlib_list())
    stdlib_names = numpy.intersect1d(lib_names, stdlib_list())

    if write_requirements:
        write_NLT(extlib_names, project_root + 'requirements.txt')

    return extlib_names, stdlib_names


def pipdeptree(regex_filter = re_startword):
    lines = cmd_run('pipdeptree').split('\r\n')
    out = []
    for l in lines:
        if regex_filter.search(l):
            out += [l]
    for item in out:
        print(item)


def pip_review(args = ''):
    print(cmd_run(f'pip-review {args}'))
