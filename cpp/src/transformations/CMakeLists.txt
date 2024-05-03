
pybind11_add_module(
    transformations
    transformations.cpp
)

target_link_libraries(
    transformations
    PRIVATE pybind11::module
    "${TORCH_LIBRARIES}"
)
