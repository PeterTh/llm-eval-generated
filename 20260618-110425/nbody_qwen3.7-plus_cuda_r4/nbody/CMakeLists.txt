cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nbody nbody.cpp)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --extended-lambda
    >
)

# Link CUDA runtime
target_link_libraries(nbody cudart)
