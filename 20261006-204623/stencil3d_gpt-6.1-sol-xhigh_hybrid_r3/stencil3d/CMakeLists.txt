cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
# CUDA 12.0-12.3 supports GCC through version 12. Honor a supplied host
# compiler, otherwise use an installed compatible GCC when the default is newer.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX}
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
   AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12
   AND CUDAToolkit_VERSION VERSION_LESS 12.4)
    find_program(STENCIL_CUDA_HOST_CXX NAMES g++-12 g++-11)
    if(STENCIL_CUDA_HOST_CXX)
        set(CMAKE_CUDA_HOST_COMPILER "${STENCIL_CUDA_HOST_CXX}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()
# Supply optimized cubins plus PTX for later GPUs; cross builds can override
# CMAKE_CUDA_ARCHITECTURES. CUDA 13 no longer accepts the Volta target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES 80 90)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 70 80)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60 70)
    endif()
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the original source and executable names while compiling CUDA kernels.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

# Imported OpenMP targets supply compile flags only for CXX sources, so forward
# those flags to nvcc's host compiler. Avoid fast math to preserve FP results.
separate_arguments(STENCIL_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS STENCIL_OPENMP_FLAGS)
    target_compile_options(stencil3d PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(stencil3d PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>")

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
