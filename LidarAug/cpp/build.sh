#!/usr/bin/bash

if [[ $1 == "clean" ]]; then
    echo "Removing build directory..."
    if [[ $2 == "verbose" ]]; then
        rm -rvf build
    else
        rm -rf build
    fi

else
    cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" -S . -B build
    cmake --build build
    # cmake --build . --config Release

    # call script with argument 'build' to only build the project, not run the tests
    if [[ $1 != "build" ]]; then
        cd build && ctest
    fi
fi
