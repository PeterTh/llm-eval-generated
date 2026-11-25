cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(detect_cuda_architectures ON)
endif()
project(stencil3d_benchmark LANGUAGES CUDA)

# Build native code and PTX for detected GPU architectures supported by the toolkit.
if(detect_cuda_architectures)
    foreach(native_arch IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
        string(REGEX REPLACE "-real$" "" arch "${native_arch}")
        if(arch IN_LIST CMAKE_CUDA_ARCHITECTURES_ALL OR native_arch IN_LIST CMAKE_CUDA_ARCHITECTURES_ALL)
            list(APPEND detected_cuda_architectures "${arch}")
        endif()
    endforeach()
    if(detected_cuda_architectures)
        set(CMAKE_CUDA_ARCHITECTURES "${detected_cuda_architectures}" CACHE STRING "CUDA architectures" FORCE)
    endif()
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)

target_compile_options(stencil3d PRIVATE -O3)
