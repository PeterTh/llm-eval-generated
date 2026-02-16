cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Enable optimizations
# Enable optimizations
set(CUDA_NVCC_FLAGS "-O3 -Xcompiler -Wall,-Wextra")

# Link math library
target_link_libraries(black_scholes m)
