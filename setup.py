from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup
import platform

link_args = ['-lz']

if platform.system() == 'Darwin':
    link_args.append("-lomp")

MODULE_NAME = "lidar_aug"

CPP_VERSION = "-std=c++17"
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
        sources=["cpp/src/weather.cpp", "cpp/src/raytracing.cpp"],
        library_dirs=['/usr/local/lib', '/usr/local/lib64/'],
        libraries=['cnpy'],
        define_macros=[("BUILD_MODULE", None)],
        extra_link_args=link_args,
        extra_compile_args=[
            CPP_VERSION, COMPILER_OPTIMIZATION_LEVEL, '-I/usr/local/include'
        ],
    ),
    CppExtension(
        name=f"{MODULE_NAME}.evaluation",
        sources=["cpp/src/evaluation.cpp", "cpp/src/utils.cpp"],
        define_macros=[("BUILD_MODULE", None)],
        extra_compile_args=[CPP_VERSION, COMPILER_OPTIMIZATION_LEVEL],
    ),
    CppExtension(name=f"{MODULE_NAME}.point_cloud",
                 sources=["cpp/src/point_cloud.cpp"],
                 define_macros=[("BUILD_MODULE", None)],
                 extra_compile_args=[CPP_VERSION,
                                     COMPILER_OPTIMIZATION_LEVEL]),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
