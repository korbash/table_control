import sys

if sys.platform == "win32":
    from .win import *
if sys.platform == "linux2":
    from .linux import *
