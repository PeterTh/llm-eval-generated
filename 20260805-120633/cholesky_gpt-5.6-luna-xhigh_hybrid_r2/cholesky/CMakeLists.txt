cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CHOL_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING
    "CUDA SM versions to compile for (override for the target cluster)")

# Keep the source file set unchanged while compiling it as CUDA so the
# executable name and benchmark layout remain compatible with the original.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CHOL_CUDA_ARCHITECTURES}"
    LINKER_LANGUAGE CUDA
)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    -O3 -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
)

target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cublas
    CUDA::cudart
    m
)
