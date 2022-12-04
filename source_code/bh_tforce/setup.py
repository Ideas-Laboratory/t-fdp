from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# GCC_INC_DIR = "/usr/lib/gcc/x86_64-linux-gnu/5/include/"

# package = Extension('bh_force', ['bh_force.pyx'], include_dirs=[numpy.get_include(),"/usr/lib/gcc/x86_64-linux-gnu/5/include/"])
package = Extension('bh_tforce', ['bh_tforce.pyx'], include_dirs=[numpy.get_include(
)], extra_compile_args=['-fopenmp'], extra_link_args=["-fopenmp"])
setup(ext_modules=cythonize([package]))
