import sys

if sys.platform == "win32":
    from driwers.win import *
if sys.platform == "linux2":
    from driwers.linux import *
