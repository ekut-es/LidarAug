
# LidarAug

TODO

## Installation

First clone and enter the repository:

`git clone https://github.com/ekut-es/LidarAug && cd LidarAug`

### C++ library

The following dependencies are necessary to build and test the C++ library for development:

- [libtorch](https://pytorch.org/get-started/locally/)
- [google test](https://github.com/google/googletest)
- [boost](https://www.boost.org/)
- GCC 13 or higher

It is also necessary to set the environment variable `TORCH_PATH` to point to where `libtorch` is installed on your system.

After that, just run `make ctest` to compile the library and run google test.

### Python library

To use the Python library, just run `make install` after cloning and entering the repository.

To test the python functions/wrappers, install [pytest](https://docs.pytest.org/en/8.0.x/) (`pip install pytest`) and run `make testpy`.

The required Python version is 3.11.

#### Modules

The library contains 4 modules:

1. **transformations:**

`transformations` contains any C++ enums, structs and functions that have bindings and are used for transformations.

2. **weather_simulations:**

`weather_simulations` contains any C++ enums, structs and functions that have bindings and are used for weather simulations.

3. **augmentations:**

`augmentations` contains the Python wrappers for any C++ function (weather simulation or transformation).

4. **evaluation:**

`evaluation` contains (C++) function to evaluate the accuracy of bounding boxes.
