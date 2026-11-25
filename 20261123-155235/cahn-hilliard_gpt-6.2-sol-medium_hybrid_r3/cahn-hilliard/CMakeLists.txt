cmake_minimum_required(VERSION 3.18)

# Include native code for major GPU generations by default, including when
# configuring on a login node. Cluster builds can target a specific GPU.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23
   AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    LINKER_LANGUAGE CXX
)
target_compile_options(cahn_hilliard PRIVATE
    -O3
    --fmad=false
    "SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}"
)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
