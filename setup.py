from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

MODULE_NAME = "LidarAug"

CPP_VERSION = "-std=c++20"
COMPILER_OPTIMIZATION_LEVEL = "-O3"

ext_modules = [
    CppExtension(
        name=f"{MODULE_NAME}.transformations",
        sources=["cpp/src/transformations.cpp", "cpp/src/tensor.cpp"],
        define_macros=[("BUILD_MODULE", None)],
        extra_compile_args=[CPP_VERSION, COMPILER_OPTIMIZATION_LEVEL],
    ),
    CppExtension(
        name=f"{MODULE_NAME}.weather_simulations",
        sources=["cpp/src/weather.cpp"],
        define_macros=[("BUILD_MODULE", None)],
        extra_compile_args=[CPP_VERSION, COMPILER_OPTIMIZATION_LEVEL],
    ),
    CppExtension(name=f"{MODULE_NAME}.evaluation",
                 sources=["cpp/src/evaluation.cpp", "cpp/src/utils.cpp"],
                 define_macros=[("BUILD_MODULE", None)],
                 extra_compile_args=[CPP_VERSION,
                                     COMPILER_OPTIMIZATION_LEVEL]),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
