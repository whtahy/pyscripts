# Copy/paste
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import subprocess

from pyscripts.hero import re_startword


def pipdeptree(regex_filter = re_startword):
    # https://stackoverflow.com/a/4760517
    raw = subprocess.run('pipdeptree', stdout = subprocess.PIPE)
    lines = raw.stdout.decode('utf-8').split('\r\n')
    out = []
    for l in lines:
        if regex_filter.search(l):
            out += [l]
    for item in out:
        print(item)


def pip_review(cmd_args = '-a'):
    # https://stackoverflow.com/a/4760517
    result = subprocess.run(f'pip-review {cmd_args}', stdout = subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
