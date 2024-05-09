
pybind11_add_module(
    transformations
    transformations.cpp
    ../tensor.cpp
)

# find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
add_library(TORCH_PYTHON_LIBRARY SHARED IMPORTED)
set_target_properties(TORCH_PYTHON_LIBRARY PROPERTIES IMPORTED_LOCATION  "/home/tom/.local/lib/python3.12/site-packages/torch/lib/libtorch_python.so")

target_link_libraries(
    transformations
    PRIVATE pybind11::module
    "${TORCH_LIBRARIES}"
    "${Boost_LIBRARIES}"
    TORCH_PYTHON_LIBRARY

)
