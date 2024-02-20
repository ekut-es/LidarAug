
all: build test

configure:
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./LidarAug/cpp/ -B ./LidarAug/cpp/build_files

configure_test:
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./LidarAug/cpp/ -B ./LidarAug/cpp/build_files

build: configure_test
	cmake --build ./LidarAug/cpp/build_files

release: configure
	cmake --build ./LidarAug/cpp/build_files --config

test: ./LidarAug/cpp/build_files
	cd ./LidarAug/cpp/build_files && ctest

rerun: ./LidarAug/cpp/build_files
	cd ./LidarAug/cpp/build_files && ctest --rerun-failed --output-on-failure

install:
	pip install .

clean: ./LidarAug/cpp/build_files
	rm -rfv ./LidarAug/cpp/build_files
