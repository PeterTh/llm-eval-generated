cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force spmv.cpp to be compiled with nvcc (CUDA compiler)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
set_target_properties(spmv PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
)
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link math library
target_link_libraries(spmv m)
