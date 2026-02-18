cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link math library
target_link_libraries(matmul PRIVATE m)
