from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("sphprojection.kernel", ["pyxhydro/sphprojection/kernel.pyx"])
]

setup(
    name='pyxhydro',
    version='1.0.0',
    package_dir={'': 'pyxhydro'},
    packages=['readgadget', 'readgadget.modules', 'pygadgetreader', 'sphprojection', 'gadgetutils', 'specutils'],
    ext_modules=cythonize(extensions, force=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    url='https://github.com/mroncarelli/pyxhydro',
    license='',
    author='Mauro Roncarelli',
    author_email='mauro.roncarelli@inaf.it',
    description='End-to-end X-ray simulator for hydrodynamics simulations'
)
