# User scripts
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt

import inspect
import os
import re
import subprocess
import time
from typing import Any, List, Tuple

import numpy

import pandas
import requests
from pyscripts.hero import re_github, re_imports
from pyscripts.knot import os_path, path_pieces
from pyscripts.scholar import read_file, scan_dir
from stdlib_list import stdlib_list


def google(
        query: str) \
        -> str:
    """
    Google query HTML.
    """
    rq = requests.get(f'https://www.google.com/search?q={query}')
    return rq.text


def package_urls(
        project_path: str) \
        -> dict:
    url_dict = {}
    package_names = read_file(
        os_path(project_path + os.sep + 'requirements.txt'))
    for nm in package_names:
        raw = google(f'github {nm}')
        url_dict[nm] = re_github.search(raw).group()
        time.sleep(0.5)
    return url_dict


def pytest_skeleton(
        module: Any) \
        -> str:
    names = [x[0] for x in inspect.getmembers(module, inspect.isfunction)]
    s = ''
    for x in names:
        s += f'def test_{x}():\n    pass\n'
    return s


def console(
        command: str) \
        -> str:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        encoding='utf-8').stdout


def pipdeptree(
        regex: str = r'^\w+') \
        -> None:
    """
    `pipdeptree` wrapper.
    """
    lines = console('pipdeptree').split('\r\n')
    regex_filter = re.compile(regex)
    out = []
    for l in lines:
        if regex_filter.search(l):
            out += [l]
    for item in out:
        print(item)


def pip_review(
        args: str = '') \
        -> None:
    """
    `pip-review` wrapper.
    """
    print(console(f'pip-review {args}'))


def requirements(
        path: str,
        project_name: str = None,
        lib_aliases=None) \
        -> Tuple[List[str], List[str]]:
    """
    Project imports.
    """
    if project_name is None:
        project_name = path_pieces(path)[-1]
    if lib_aliases is None:
        lib_aliases = {'sklearn': 'scikit-learn'}

    file_list = scan_dir(path, file_filter=lambda x: x.endswith('.py'))

    import_statements = []
    for file in file_list:
        import_statements += read_file(
            file,
            line_filter=is_import,
            break_condition=lambda x: x.startswith('def '))

    names = []
    for line in import_statements:
        re_match = re_imports.search(line)
        if re_match:
            name = re_match.group(2)
            names += [lib_aliases.get(name, name)]

    names = pandas.unique(names)
    names = names[names != project_name]
    extlib_names = sorted(numpy.setdiff1d(names, stdlib_list()))
    stdlib_names = sorted(numpy.intersect1d(names, stdlib_list()))

    return extlib_names, stdlib_names


#
# Helpers

def is_import(
        x: str) \
        -> bool:
    """
    Whether line contains an import statement.
    """
    return x.startswith('import ') or x.startswith('from ') and ' .' not in x
