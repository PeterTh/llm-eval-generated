cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake to compile .cpp files with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Enable optimizations - separate host and device flags
target_compile_options(matmul PRIVATE --expt-relaxed-constexpr -O3 -arch=native -Xcompiler=-O3,-Wall,-Wextra)

# Link math library
target_link_libraries(matmul m)
