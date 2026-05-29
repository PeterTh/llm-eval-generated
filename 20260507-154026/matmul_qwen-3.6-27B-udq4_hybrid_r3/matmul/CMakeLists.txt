cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Build with nvcc so CUDA device code is compiled
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force nvcc to compile the .cpp file (it contains CUDA device code)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(matmul PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp
        -arch=native
    >
)

# Link MPI, OpenMP, CUDA runtime, and math
target_link_libraries(matmul
    ${MPI_CXX_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    -fopenmp
    m
)
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
