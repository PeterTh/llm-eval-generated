cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP and math library
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(stencil3d PRIVATE OpenMP::OpenMP_CXX m)
else()
    # Fallback for older CMake/toolchains
    target_compile_options(stencil3d PRIVATE -fopenmp)
    target_link_libraries(stencil3d PRIVATE m)
endif()
