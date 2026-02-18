cmake_minimum_required(VERSION 3.20)
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Make sure nbody.cpp is compiled as CUDA
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations
# For CUDA files, options are passed to nvcc. 
# Host compiler options should be passed via -Xcompiler or using generator expressions.
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(nbody PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall,-Wextra,-march=native,-fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    )
else()
    target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
