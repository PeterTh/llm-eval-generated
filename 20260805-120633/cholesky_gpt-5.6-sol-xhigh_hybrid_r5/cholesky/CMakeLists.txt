cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The benchmark is intentionally hybrid: there is no sequential or CPU-only
# fallback.  All three programming models are therefore required at configure
# time and used on every execution, including a single-rank launch.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

target_compile_options(cholesky PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    m
)
