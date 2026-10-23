cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)

find_package(MPI REQUIRED COMPONENTS CXX)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Host side: OpenMP + aggressive optimization.  Device side: optimized CUDA.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>)

# Depending on the CMake/CUDA toolchain, the final CUDA link may be driven by
# nvcc (needs -Xcompiler) or directly by the host linker (needs -fopenmp).
if(NOT CMAKE_CUDA_HOST_LINK_LAUNCHER OR CMAKE_CUDA_HOST_LINK_LAUNCHER MATCHES "nvcc")
    target_link_options(cahn_hilliard PRIVATE -Xcompiler=-fopenmp)
else()
    target_link_options(cahn_hilliard PRIVATE -fopenmp)
endif()

target_link_libraries(cahn_hilliard PRIVATE ${MPI_CXX_LIBRARIES} m)
