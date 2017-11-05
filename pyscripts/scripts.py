# Command line scripts
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import subprocess

from pyscripts.hero import re_startword


def run_cmd(command):
    # https://stackoverflow.com/a/4760517
    raw = subprocess.run(command, stdout = subprocess.PIPE).stdout
    return raw.decode('utf-8')


def pipdeptree(regex_filter = re_startword):
    lines = run_cmd('pipdeptree').split('\r\n')
    out = []
    for l in lines:
        if regex_filter.search(l):
            out += [l]
    for item in out:
        print(item)


def pip_review(args = '-a'):
    print(run_cmd(f'pip-review {args}'))
