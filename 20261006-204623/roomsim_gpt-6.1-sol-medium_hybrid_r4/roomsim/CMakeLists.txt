cmake_minimum_required(VERSION 3.18)
# Provide native binaries across supported GPU generations instead of relying
# on the compiler's old default architecture and runtime PTX compilation.
# Cluster builds can narrow this with CMAKE_CUDA_ARCHITECTURES or CUDAARCHS.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source with nvcc; keep the public executable name.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-ffp-contract=off,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
