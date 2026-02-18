cmake_minimum_required(VERSION 3.17)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

add_executable(qtclustering qtclustering.cu)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(qtclustering m)
