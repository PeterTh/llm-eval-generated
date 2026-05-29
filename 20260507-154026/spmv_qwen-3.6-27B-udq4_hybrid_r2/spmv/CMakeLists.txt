cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA separate compilation for large kernels
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Force compilation of spmv.cpp with nvcc (CUDA compiler) so CUDA kernel
# definitions and runtime API are recognized
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(spmv PROPERTIES
    LINKER_LANGUAGE CXX
)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-fopenmp,-Wall,-Wextra>
)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(spmv
    MPI::MPI_CXX
    ${OpenMP_CXX_LIBRARIES}
    m
)

# Include directories
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_PATH})
