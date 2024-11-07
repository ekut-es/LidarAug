ifeq ($(OS),Windows_NT)
    CXXFLAGS += openmp
    CFLAGS+= openmp
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		CXXFLAGS += -Xpreprocessor
		CFLAGS += -Xpreprocessor
	endif
	CXXFLAGS += -fopenmp
	CFLAGS += -fopenmp
endif

all: install testpy build ctest

configure:
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./cpp/ -B ./cpp/build_files

configure_test:
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S ./cpp/ -B ./cpp/build_files

build: configure_test
	cmake --build ./cpp/build_files -j 8

release: configure
	cmake --build ./cpp/build_files -j 8

ctest: build
	cd ./cpp/build_files && ctest

testc: build
	cd ./cpp/build_files && ./transformations_test $(ARGS)

testpy: ./pytest/test.py
	python3.11 -m pytest ./pytest/test.py -v

rerun: ./cpp/build_files
	cd ./cpp/build_files && ctest --rerun-failed --output-on-failure

sim: release
	cd ./cpp/build_files && ctest --output-on-failure -R 'Simulation.*'

install:
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "CFLAGS: $(CFLAGS)"
	rm -rf ./build ./src/lidar_aug.egg-info && mkdir -p ./tmp && TMPDIR=./tmp python3.11 -m pip install -v . && rm -rf ./tmp

docker:
	docker build -t lidar_aug:0.0.1 .

clean: ./cpp/build_files
	rm -rfv ./cpp/build_files
