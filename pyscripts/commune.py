# Shared values
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *

# Package info
PROJECT_NAME: str = 'pyscripts'
LICENSE_DIR: str = 'licenses'

# 1680 x 1050 resolution
TERM_WIDTH: int = 180
PYPLOT_WIDTH: int = 26
PYPLOT_HEIGHT: int = 16

# RNG seed
SEED: int = 123
