# User scripts
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt
import re
import subprocess
import time
from typing import TYPE_CHECKING

import numpy
import pandas
import requests
from stdlib_list import stdlib_list

from pyscripts.hero import re_github, re_imports
from pyscripts.knot import os_path, path_pieces
from pyscripts.scholar import (
    pickle_obj, read_file, scan_dir
)

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def google(
        query: str) \
        -> str:
    rq = requests.get(f'https://www.google.com/search?q={query}')
    return rq.text


def package_urls(
        project_path: str) \
        -> dict:
    url_dict = {}
    package_names = read_file(os_path(project_path + '/' + 'requirements.txt'))
    for nm in package_names:
        raw = google(f'github {nm}')
        url_dict[nm] = re_github.search(raw).group()
        time.sleep(0.5)
    pickle_obj(url_dict, os_path(project_path + '/' + 'url_dict.pickle'))
    return url_dict


def console(
        command: str) \
        -> str:
    # https://stackoverflow.com/a/4760517
    raw = subprocess.run(command, stdout = subprocess.PIPE).stdout
    return raw.decode('utf-8')


def lib_names(
        path: str,
        project_name: str = None,
        lib_aliases = None) \
        -> 'Tuple[List[str], List[str]]':
    if project_name is None:
        project_name = path_pieces(path)[-1]
    if lib_aliases is None:
        lib_aliases = {'sklearn': 'scikit-learn'}

    file_list = scan_dir(path, file_filter = lambda x: x.endswith('.py'))

    import_statements = []
    for file in file_list:
        import_statements += read_file(
                file,
                line_filter = lambda x: x.startswith('import ')
                                        or x.startswith('from ')
                                        and ' .' not in x,
                break_condition = lambda x: x.startswith('def '))

    # print_each(import_statements)

    names = []
    for line in import_statements:
        re_match = re_imports.search(line)
        if re_match:
            name = re_match.group(2)
            names += [lib_aliases.get(name, name)]

    # print_each(names)

    names = pandas.unique(names)
    names = names[names != project_name]
    extlib_names = sorted(numpy.setdiff1d(names, stdlib_list()))
    stdlib_names = sorted(numpy.intersect1d(names, stdlib_list()))

    return extlib_names, stdlib_names


def pipdeptree(
        regex: str = r'^\w+') \
        -> None:
    lines = console('pipdeptree').split('\r\n')
    regex_filter = re.compile(regex)
    out = []
    for l in lines:
        if regex_filter.search(l):
            out += [l]
    for item in out:
        print(item)


def pip_review(
        args = '') \
        -> None:
    print(console(f'pip-review {args}'))
