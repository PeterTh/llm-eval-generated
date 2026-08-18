cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing filename and executable name; compile the translation unit
# with nvcc because it now contains the CUDA kernels and their host driver.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Permit an explicit cluster-specific value (-DCMAKE_CUDA_ARCHITECTURES=...),
# otherwise emit optimized code for every architecture supported by the CUDA
# toolkit in use. Version guards keep the default usable with older toolkits.
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(_nbody_cuda_architectures 60 70 75)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        list(APPEND _nbody_cuda_architectures 80)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        list(APPEND _nbody_cuda_architectures 86)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND _nbody_cuda_architectures 89 90)
    endif()
    set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES ${_nbody_cuda_architectures})
endif()
