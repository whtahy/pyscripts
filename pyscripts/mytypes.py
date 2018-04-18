# Mypy static type aliases
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import Union

import numpy as np

Numeric = Union[int, float, np.number]
