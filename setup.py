#from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    CppExtension(name="transformations", sources=["LidarAug/transformations.cpp"]),
]

setup(
    name='LidarAug',
    version=__version__,
    install_requires=[
        'torch',
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
