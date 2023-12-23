
all: build test

configure:
	cmake -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S . -B build_files

configure_test:
	cmake -DCMAKE_PREFIX_PATH="$(TORCH_PATH)" -S . -B build_files

build: configure_test
	cmake --build build_files

release: configure
	cmake --build build_files --config

test: build_files
	cd build_files && ctest

rerun: build_files
	cd build_files && ctest --rerun-failed --output-on-failure

install:
	cd ../.. && pip install .

clean: build_files
	rm -rfv build_files
