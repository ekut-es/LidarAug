Quickstart
==========

First clone and enter the repository:

``git clone https://github.com/ekut-es/LidarAug && cd LidarAug``

Local Installation
------------------

To install the module locally you need to first install the dependencies.

- `PyTorch/libtorch <https://pytorch.org/get-started/locally/>`__
- `boost <https://www.boost.org/>`__
- `OpenMP <https://www.openmp.org/resources/openmp-compilers-tools/>`__
- `cnpy <https://github.com/TomSchammo/cnpy>`__
- `pybind11 <https://github.com/pybind/pybind11>`__

Easiest way is to run:

``python3.11 -m pip install -r requirements.txt``

And install `boost <https://www.boost.org/>`__, `OpenMP <https://www.openmp.org/resources/openmp-compilers-tools/>`__ and `cnpy <https://github.com/TomSchammo/cnpy>`__ by following the installation instructions of the respective project.

After installing all the necessary dependencies run ``make install``.

**Optional:** You can test the installation by running ``make testpy``.

Docker
------

Alternatively you can use `Docker <https://www.docker.com/>`__.

To build the Docker image simply run ``make docker``.

This will create a Docker image with the module and all the necessary dependencies.

You can run it using ``docker run -it lidar_aug:1.0.1 bash``

**NOTE:** For those using ARM run ``make docker-arm`` instead to build the docker image.
