cmake_minimum_required(VERSION 3.23)
# Build on login nodes as well as GPU nodes. A cluster build can select its
# specific architecture with CMAKE_CUDA_ARCHITECTURES or CUDAARCHS.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(roomsim PRIVATE -O3 --fmad=false
    -Xcompiler=${OpenMP_CXX_FLAGS} -Xcompiler=-ffp-contract=off)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
