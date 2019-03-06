from distutils.core import setup
from distutils.extension import Extension

# Obtain the numpy include directory.  This logic works across numpy versions.
import numpy as np
from Cython.Distutils import build_ext

try:

    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "nmsclib",
        ["cpu_nms.pyx"],
        include_dirs = [numpy_include],
	    extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    )
]
setup(
    name='yolonms',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
