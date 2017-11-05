# Copy/paste
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import subprocess

from pyscripts.hero import re_startword


def pipdeptree():
    # https://stackoverflow.com/a/4760517
    cmd_out = subprocess.run('pipdeptree', stdout = subprocess.PIPE)
    package_names = []
    for line in cmd_out.stdout.decode('utf-8').split('\r\n'):
        if re_startword.search(line):
            package_names += [line]
    for nm in package_names:
        print(nm)


def pip_review():
    # https://stackoverflow.com/a/4760517
    result = subprocess.run('pip-review -a', stdout = subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
