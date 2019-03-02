import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import  cythonize

setup(
    name='cpu_nms1',
    ext_modules=cythonize([Extension('cpu_nms',
                                     ['cpu_nms.pyx'],
                                     include_dirs=[np.get_include()],

                                     )

                           ])
)