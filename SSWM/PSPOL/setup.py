from distutils.core import setup
from Cython.Build import cythonize

ext_modules = cythonize("pspol.pyx", annotate=True)

for e in ext_modules:
    e.cython_directives = {"boundscheck": False,
                           "wraparound" : False,
                           "cdivision" : True}
    
setup(
    ext_modules = ext_modules
)
