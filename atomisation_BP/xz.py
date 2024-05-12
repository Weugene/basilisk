from __future__ import annotations

import imp
from os.path import expanduser
from sys import stderr
from sys import stdout

from bview import BCanvas
from Tkinter import mainloop
from Tkinter import Tk

# Try user customization
try:
    imp.load_source("userinit", expanduser("~/.bview.py"))
except OSError:
    pass

root = Tk()
root.withdraw()
root.title("bview")
root.canvas = BCanvas(root, stdout)
stdout = stderr

try:
    mainloop()
except (KeyboardInterrupt, SystemExit):
    None
