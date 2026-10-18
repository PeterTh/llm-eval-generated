cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-ffp-contract=off,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
