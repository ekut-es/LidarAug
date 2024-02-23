
all: build test

configure:
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./cpp/ -B ./cpp/build_files

configure_test:
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./cpp/ -B ./cpp/build_files

build: configure_test
	cmake --build ./cpp/build_files -j 4

release: configure
	cmake --build ./cpp/build_files -j 4 --config

test: ./cpp/build_files
	cd ./cpp/build_files && ctest

rerun: ./cpp/build_files
	cd ./cpp/build_files && ctest --rerun-failed --output-on-failure

install:
	rm -rf ./build ./src/LidarAug.egg-info && mkdir -p ./tmp && TMPDIR=./tmp pip install . && rm -rf ./tmp

clean: ./cpp/build_files
	rm -rfv ./cpp/build_files
