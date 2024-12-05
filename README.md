# LidarAug

A toolbox for LiDAR point cloud data, providing point cloud transformations, point cloud augmentation, realistic weather simulation and 2D & 3D AP evaluation, with an easy-to-use Python API.
This module supports several essential tasks for the development of LiDAR-based perception methods in automated driving.

## Installation

First clone and enter the repository:

`git clone https://github.com/ekut-es/LidarAug && cd LidarAug`

### C++ library

The following dependencies are necessary to build and test the C++ library for development:

- [libtorch](https://pytorch.org/get-started/locally/)
- [google test](https://github.com/google/googletest)
- [boost](https://www.boost.org/)
- [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/)

It is also necessary to set the environment variable `TORCH_PATH` to point to where `libtorch` is installed on your
system.

After that, just run `make ctest` to compile the library and run google test.

*Note that the tests written for the backend include some controlled RNG tests which might fail on different platforms with different architectures such as the Apple MX chips. The tests were developed for Linux x86 using GCC.*

### Python module

The following dependencies are necessary to install the Python module:

- [PyTorch/libtorch](https://pytorch.org/get-started/locally/)
- [boost](https://www.boost.org/)
- [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/)
- [cnpy](https://github.com/TomSchammo/cnpy)
- [pybind11](https://github.com/pybind/pybind11)

To use the Python module, just run `make install` after cloning and entering the repository.

To test the python functions/wrappers, install [pytest](https://docs.pytest.org/en/8.0.x/) (`pip install pytest`) and
run `make testpy`.

The required Python version is 3.11.

#### Submodules

The `lidar_aug` Python module contains 5 submodules:

1. **transformations:**

`transformations` contains any C++ enums, structs and functions that have bindings and are used for transformations.

2. **weather_simulations:**

`weather_simulations` contains any C++ enums, structs and functions that have bindings and are used for weather
simulations.

3. **augmentations:**

`augmentations` contains the Python wrappers for any C++ function (weather simulation or transformation).

4. **evaluation:**

`evaluation` contains (C++) function to evaluate the accuracy of bounding boxes.
This can be done for 2D and 3D bounding boxes.

5. **point_cloud:**

`point_cloud` contains things that is specific to point clouds that is used across modules and functionally not
specific to the task of one of those.
Such as the `IntensityRange` enum that is used to set/determine the maximum intensity of the points in a point cloud.


### Docker

Alternatively the module can be run inside a [Docker](https://www.docker.com/) container.

After installing [Docker](https://www.docker.com/) and cloning the repository, all you need to do is run `make docker`,
which will start building the image and automatically run the tests during the build process.

NOTE: If you're running the docker image on ARM run `make docker-arm` instead.
