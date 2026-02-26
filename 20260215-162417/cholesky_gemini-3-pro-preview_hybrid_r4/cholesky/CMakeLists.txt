cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 -arch=native -Xcompiler -fopenmp>)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Link libraries
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# If CUDA is found, add it
enable_language(CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
