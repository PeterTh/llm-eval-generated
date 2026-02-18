cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to enable CUDA if available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA compiler found and enabled")
else()
    message(STATUS "CUDA compiler not found, will compile CPU-only version")
endif()

add_subdirectory(floydwarshall)


