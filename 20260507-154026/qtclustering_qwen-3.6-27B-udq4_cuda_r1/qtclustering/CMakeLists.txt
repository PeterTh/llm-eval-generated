cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture flags for maximum performance
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures" FORCE)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

# Link math library
target_link_libraries(qtclustering m)
