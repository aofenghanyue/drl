
import os

# add mingw64/bin to PATH
os.add_dll_directory(os.environ['minGWPath'])

from build import pybindDemo

print(pybindDemo.add(1, 2))