cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_70,code=sm_70>)

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX 
    m 
    CUDA::cudart 
    CUDA::cublas 
    CUDA::cusolver
)

# Treat .cpp as CUDA code if necessary, or just rely on nvcc handling it if we use .cu extension in a real scenario.
# But since we can't rename, we use set_source_files_properties.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
