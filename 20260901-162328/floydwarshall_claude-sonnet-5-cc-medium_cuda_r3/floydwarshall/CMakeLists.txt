cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

add_executable(floydwarshall floydwarshall.cpp)

# Compile as CUDA so host and device code (kernels, <<<>>> launches) can live
# in the single existing .cpp source file.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-Wall,-Wextra>
)

# Link math library
target_link_libraries(floydwarshall m)
