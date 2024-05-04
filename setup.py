from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup

MODULE_NAME = "LidarAug"

ext_modules = [
    CppExtension(
        name=f"{MODULE_NAME}.transformations",
        sources=[
            "cpp/src/transformations/transformations.cpp", "cpp/src/tensor.cpp"
        ],
        define_macros=[("BUILD_MODULE", None)],
    ),
    CppExtension(
        name=f"{MODULE_NAME}.weather_simulations",
        sources=["cpp/src/weather_simulations/weather.cpp"],
        define_macros=[("BUILD_MODULE", None)],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
