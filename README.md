
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

After that, just run `make` to compile the library and run the tests.

### Python library

For the Python library, just run `make install` after cloning and entering the repository.
~Currently, this does not work with Python 3.12, as there is no official release for [torch](https://github.com/pytorch/pytorch/issues/110436) yet.~

~The current nightly build of `torch` works, but there is no all around support for the tool chain yet.~

#### Modules

The library contains 3 modules:

1. **transformations:**

`transformations` contains any C++ enums, structs and functions that have bindings and are used for transformations.

2. **weather_simulations:**

`weather_simulations` contains any C++ enums, structs and functions that have bindings and are used for weather simulations.

3. **augmentations:**

`augmentations` contains the Python wrappers for any C++ function (weather simulation or transformation).
