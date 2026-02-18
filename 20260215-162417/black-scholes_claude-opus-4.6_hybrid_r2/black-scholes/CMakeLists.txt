cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp as CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# CUDA architectures
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compile flags
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-O3>
)

# Include and link
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(black_scholes m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
