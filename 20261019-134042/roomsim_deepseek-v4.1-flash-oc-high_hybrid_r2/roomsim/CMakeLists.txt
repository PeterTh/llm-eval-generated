cmake_minimum_required(VERSION 3.20)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)

# The single translation unit contains CUDA kernels, so compile it with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings for both the host and device compilation.
# OpenMP flags must be forwarded to the nvcc host compiler.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra;-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic;${OpenMP_CXX_FLAGS}>)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} m)
