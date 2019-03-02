import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import  cythonize
setup(
    name='gg',
    ext_modules= cythonize([Extension('cython_anchor',
                ['cython_anchor.pyx'],
                include_dirs=[np.get_include()]
                           )]),
    cmdclass = { 'build_ext':build_ext}
)