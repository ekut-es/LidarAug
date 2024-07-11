from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

MODULE_NAME = "LidarAug"

ext_modules = [
    CppExtension(
        name=f"{MODULE_NAME}.transformations",
        sources=["cpp/src/transformations.cpp", "cpp/src/tensor.cpp"],
        define_macros=[("BUILD_MODULE", None)],
        extra_compile_args=['-std=c++20', '-O3'],
    ),
    CppExtension(
        name=f"{MODULE_NAME}.weather_simulations",
        sources=["cpp/src/weather.cpp", "cpp/src/raytracing.cpp"],
        define_macros=[("BUILD_MODULE", None)],
        extra_compile_args=['-std=c++20', '-O3'],
    ),
    CppExtension(name=f"{MODULE_NAME}.evaluation",
                 sources=["cpp/src/evaluation.cpp", "cpp/src/utils.cpp"],
                 define_macros=[("BUILD_MODULE", None)],
                 extra_compile_args=['-std=c++20', '-O3']),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
