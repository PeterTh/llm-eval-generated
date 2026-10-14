cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on the machine
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(Threads REQUIRED)

add_executable(matmul matmul.cpp)

# matmul.cpp is compiled by nvcc so the kernel can live in the same file
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# -fmad=false keeps each `sum += a * b` as a separate rounded multiply and add,
# matching the sequential reference's floating-point results.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-march=native,-Wall,-Wextra>)

# Link math library and threads (one host thread per GPU)
target_link_libraries(matmul m Threads::Threads)
