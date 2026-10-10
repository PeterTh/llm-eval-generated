cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Older CUDA 12 releases reject GCC 13. Honor explicit toolchains, otherwise
# use an installed GCC 12 for those releases when it is available.
if(UNIX AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    if(DEFINED CMAKE_CUDA_COMPILER)
        set(_cahn_nvcc "${CMAKE_CUDA_COMPILER}")
    elseif(DEFINED ENV{CUDACXX})
        set(_cahn_nvcc "$ENV{CUDACXX}")
    else()
        find_program(_cahn_nvcc NAMES nvcc)
    endif()
    if(_cahn_nvcc)
        execute_process(COMMAND "${_cahn_nvcc}" --version OUTPUT_VARIABLE _cahn_cuda_version)
        if(_cahn_cuda_version MATCHES "release 12\\.[0-3],")
            find_program(_cahn_cuda_host NAMES g++-12)
            if(_cahn_cuda_host)
                set(CMAKE_CUDA_HOST_COMPILER "${_cahn_cuda_host}")
            endif()
        endif()
    endif()
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Preserve the double-precision arithmetic and its evaluation order.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>
)
