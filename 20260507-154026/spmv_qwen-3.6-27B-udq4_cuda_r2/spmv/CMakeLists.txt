cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Mark source as CUDA despite .cpp extension
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations - use native GPU architecture
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra --expt-relaxed-constexpr)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(spmv m)
