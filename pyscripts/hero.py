# Regular expressions
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import re

re_imports = re.compile(r'^\s*(from|import)\s+\w+')
re_typename = re.compile(r'([a-zA-Z]+)(\d*).*$')
re_ext = re.compile(r'.+(\.\w+)$')
re_slash = re.compile(r'[\\/]+')
