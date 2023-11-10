from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup

ext_modules = [
    CppExtension(name="LidarAug.transformations",
                 sources=["LidarAug/cpp/src/transformations.cpp"]),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
