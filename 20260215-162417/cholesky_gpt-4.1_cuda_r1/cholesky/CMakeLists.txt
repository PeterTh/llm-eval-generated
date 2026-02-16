cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
cuda_add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(cholesky m)
