cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lidar_aug)

# GoogleTest requires at least C++14, but torch requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find clang-tidy
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if (CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" -checks=-*,readability-*,clang-analyzer-*,boost-*,cppcoreguidelines-*,bugprone-*,modernize-*,-modernize-pass-by-value,-modernize-use-trailing-return-type,-bugprone-easily-swappable-parameters,-bugprone-suspicious-include,-readability-identifier-length)
else ()
    message(STATUS "clang-tidy not found")
endif ()


add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if (MSVC)
    # additional warning levels for msvc (commonly used on windows)
    add_compile_options(/W4)
else ()
    # additional warnings for other compilers
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
endif ()

# Add sign conversion warnings
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wconversion)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wsign-conversion)
endif ()

# NOTE(tom): Add this to run the tests that require controlled RNG
add_compile_definitions(TEST_RNG)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(FetchContent)
FetchContent_Declare(cnpy
        GIT_REPOSITORY "https://github.com/TomSchammo/cnpy"
        GIT_TAG cpp17
        GIT_SHALLOW True
)
FetchContent_MakeAvailable(cnpy)


find_package(Boost REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenMP)

# Testing code
enable_testing()

add_executable(
        transformations_test
        src/transformations.cpp
        include/transformations.hpp
        include/weather.hpp
        src/weather.cpp
        include/tensor.hpp
        src/tensor.cpp
        test/test.cpp
        include/utils.hpp
        src/utils.cpp
        include/evaluation.hpp
        src/evaluation.cpp
        include/raytracing.hpp
        src/raytracing.cpp
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(
            transformations_test
            cnpy
            "${TORCH_LIBRARIES}"
            "${Boost_LIBRARIES}"
            OpenMP::OpenMP_CXX
            "${ZLIB_LIBRARIES}"
            GTest::gtest_main
            # GTest::gtest
    )

else ()
    target_link_libraries(
            transformations_test
            "${TORCH_LIBRARIES}"
            "${Boost_LIBRARIES}"
            "${ZLIB_LIBRARIES}"
            GTest::gtest_main
            cnpy::cnpy
            # GTest::gtest
    )
endif ()


# set_property(TARGET transformations_test PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example>)
endif (MSVC)

include(GoogleTest)
gtest_discover_tests(transformations_test)
