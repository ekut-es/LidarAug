
# FROM ubuntu:22.04
FROM ursamajorlab/jammy-python:3.11

RUN apt-get update
RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update

# Verify pip installation
RUN python3.11 -m pip --version
RUN python3.11 -m pip install --upgrade pip

# Install C++ build tools & git
RUN apt-get install -y build-essential git gcc-13 g++-13 cmake libstdc++-13-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Update alternatives for GCC
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 50

# Install Boost
RUN apt-get install -y libboost-all-dev

# Install OpenMP
RUN apt-get install -y libomp-dev

# Install cnpy
RUN git clone https://github.com/TomSchammo/cnpy
RUN cd cnpy && mkdir build && cd build && cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" && make VERBOSE=1 && make install

# Install setuptools
RUN python3.11 -m pip install setuptools==69.5.1

# Install dependencies
RUN python3.11 -m pip install pytest pybind11 numpy==1.24.0 scipy
# RUN python3.11 -m pip install torch==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.11 -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu


# Set CXX flags
ENV CXXFLAGS="-fopenmp"
ENV CFLAGS="-fopenmp"

# Copy the source/project files
COPY . /opt/LidarAug

# Set the working directory
WORKDIR /opt/LidarAug

# Build the library using cmake and run the c++ tests
RUN export TORCH_PATH=$(python3.11 -c 'import torch; import os; print(os.path.join(torch.__path__[0], "share", "cmake"))'); make clean ctest

# Build and install the python module
RUN make install

# Test the python module
# RUN make testpy

# Provide entry point (I don't know if I should keep this)
CMD [ "make", "testpy" ]
