cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the source as CUDA so it can contain both host (MPI/OpenMP) and
# device (CUDA kernel) code in a single translation unit.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp --extended-lambda>
)

target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
