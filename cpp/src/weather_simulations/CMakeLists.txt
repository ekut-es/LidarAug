
pybind11_add_module(
    weather_simulations
    weather.cpp
)

target_link_libraries(
    weather_simulations
    PRIVATE pybind11::module
    "${TORCH_LIBRARIES}"
)
