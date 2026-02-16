cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Treat .cpp as CUDA source so nvcc compiles it (hybrid CUDA/C++)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>)

# Link MPI and OpenMP
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Ensure include directories from MPI are used
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH})
