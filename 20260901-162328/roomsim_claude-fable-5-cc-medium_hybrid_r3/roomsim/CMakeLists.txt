cmake_minimum_required(VERSION 3.20)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "86")
endif()

# Enable optimizations and warnings (host flags forwarded through nvcc)
target_compile_options(roomsim PRIVATE
    -O3 --expt-relaxed-constexpr
    -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math library, MPI, and OpenMP
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
