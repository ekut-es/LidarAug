
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

It is also necessary to set the environment variable `TORCH_PATH` to point to where `libtorch` is installed on your system.

After that, just run `make ctest` to compile the library and run google test.

### Python library

To use the Python library, just run `make install` after cloning and entering the repository.
Currently, this branch does not work with Python 3.12, as there is no official release for [open3d](https://github.com/isl-org/Open3D/issues/6433) yet, which is used for visualization.
Open3d is waiting for the full release of TensorFlow 2.16.0, which adds Python 3.12 support.

On Linux at least TensorFlow 2.16.0-rc0 is available, but open3d is waiting for a full release before adding 3.12 support.

To test the python functions/wrappers, install [pytest](https://docs.pytest.org/en/8.0.x/) (`pip install pytest`) and run `make testpy`.

#### Modules

The library contains 3 modules:

1. **transformations:**

`transformations` contains any C++ enums, structs and functions that have bindings and are used for transformations.

2. **weather_simulations:**

`weather_simulations` contains any C++ enums, structs and functions that have bindings and are used for weather simulations.

3. **augmentations:**

`augmentations` contains the Python wrappers for any C++ function (weather simulation or transformation).
