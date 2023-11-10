
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" ..
cmake --build . --config Release
