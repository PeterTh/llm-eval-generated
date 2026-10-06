cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
# -ffp-contract=off keeps the kernel's explicit mul/add rounding (FMA only where requested)
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link OpenMP and math library
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX m)
