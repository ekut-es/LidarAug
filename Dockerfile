
FROM ubuntu:22.04 AS base

ARG RUN_CTEST="false"
ARG RUN_PYTEST="true"
ARG OPENSSL_VERSION="3.3.2"
ARG PYTHON_VERSION="3.11.0"


RUN apt-get update && apt-get dist-upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

FROM base AS python_builder
RUN apt-get install -y \
  build-essential \
  libbz2-dev \
  libffi-dev \
  libgdbm-dev \
  liblzma-dev \
  libncurses5-dev \
  libnss3-dev \
  libsqlite3-dev \
  wget \
  zlib1g-dev

RUN echo "Downloading sources..."

RUN \
  wget -cq https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -O - | tar -xz && \
  wget -cq https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz -O - | tar -xz

RUN echo "Building OpenSSL ${OPENSSL_VERSION}..."
WORKDIR /openssl-${OPENSSL_VERSION}

# RPATH from https://wiki.openssl.org/index.php/Compilation_and_Installation#Using_RPATHs
RUN ./config -Wl,-rpath=/usr/local/ssl/lib64:/usr/local/lib -Wl,--enable-new-dtags --prefix=/usr/local/ssl --openssldir=/usr/local/ssl
RUN make
RUN make install

RUN echo "Building Python ${PYTHON_VERSION}"
WORKDIR /Python-${PYTHON_VERSION}

# Setting env variables for python compilation
ENV LDFLAGS="-L/usr/local/ssl/lib64/ -Wl,-rpath=/usr/local/ssl/lib64:/usr/local/lib"
ENV LD_LIBRARY_PATH="/usr/local/ssl/lib/:/usr/local/ssl/lib64/"
ENV CPPFLAGS="-I/usr/local/ssl/include -I/usr/local/ssl/include/openssl"

RUN \
  ./configure \
  --with-openssl=/usr/local/ssl \
  --enable-loadable-sqlite-extensions \
  --enable-shared \
  --with-openssl-rpath=auto \
  --enable-optimizations --with-lto

RUN make
RUN make altinstall

# taken from the official image https://github.com/docker-library/python/
RUN find /usr/local -depth \
  \( \
  \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
  -o \
  \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
  \) -exec rm -rf '{}' +;

# Cleaning up openSSL headers and doc
RUN rm -rf /usr/local/ssl/share /usr/local/ssl/include

FROM base AS python_base
RUN apt-get install -y --no-install-recommends \
  bzip2 \
  ca-certificates \
  curl \
  libffi8 \
  libgdbm6 \
  liblzma5 \
  libncurses6 \
  libnss3 \
  sqlite3 \
  wget \
  zlib1g \
  && apt-get autoclean \
  && apt-get clean \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

COPY --from=python_builder /usr/local/ /usr/local/

ARG PYTHON_VERSION
ENV LD_LIBRARY_PATH="/usr/local/ssl/lib/:/usr/local/ssl/lib64/"
RUN echo "Python ${PYTHON_VERSION} has been successfully installed!"

# OpenSSL looks into $OPENSSLDIR/certs as CA trust store. By default this is
# empty, and installing ca-certificates with apt-get populates it in the system
# openssl at /usr/lib/ssl/certs/. Our compiled openssl looks into
# /usr/local/ssl/certs, we create a symlink between the two to let Python access
# the OS trust store.
RUN rm -rf /usr/local/ssl/certs && ln -s /usr/lib/ssl/certs/ /usr/local/ssl/certs

WORKDIR /usr/local/bin/

RUN ln -sf python${PYTHON_VERSION%.*} python && ln -sf python${PYTHON_VERSION%.*} python3

# Verify pip installation
RUN python3.11 -m pip --version
RUN python3.11 -m pip install --upgrade pip

FROM python_base AS module_builder

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update

# Install C++ build tools & git
RUN apt-get install -y build-essential git gcc-13 g++-13 cmake libstdc++-13-dev
RUN apt-get install -y libgl1

# Update alternatives for GCC
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 50

# Install Boost
RUN apt-get install -y libboost-all-dev

# Install OpenMP
RUN apt-get install -y libomp-dev

# Install cnpy
RUN git clone https://github.com/TomSchammo/cnpy /opt/cnpy
RUN mkdir /opt/cnpy/build
WORKDIR /opt/cnpy/build
RUN cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
RUN make VERBOSE=1
RUN make install

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

RUN make clean

# Build the library using cmake and run the c++ tests
RUN if [ "${RUN_CTEST}" = "true" ]; then export TORCH_PATH=$(python3.11 -c 'import torch; import os; print(os.path.join(torch.__path__[0], "share", "cmake"))'); make ctest; fi

# Build and install the python module
RUN make install

# Test the python module
RUN if [ "${RUN_PYTEST}" = "true" ]; then make testpy; fi

# Clean up
RUN python3.11 -m pip uninstall --yes pybind11 pytest

FROM python_base AS final

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update
RUN apt-get install --only-upgrade -y libstdc++6
RUN apt-get install -y \
  libgomp1 \
  libx11-6 \
  libgl1

RUN python3.11 --version

COPY --from=module_builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

RUN python3.11 -c 'import torch; import LidarAug'

CMD ["tail", "-f", "/dev/null"]
