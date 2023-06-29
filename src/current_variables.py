# print how much space each variable requires
from __future__ import print_function  # for Python2

import sys

local_vars = list(locals().items())
for var, obj in local_vars:
    print(var, sys.getsizeof(obj))