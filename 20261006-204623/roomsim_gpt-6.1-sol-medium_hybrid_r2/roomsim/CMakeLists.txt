cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA, preserving the executable and file names.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
# Preserve the original floating-point accumulation order and accurate division.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
