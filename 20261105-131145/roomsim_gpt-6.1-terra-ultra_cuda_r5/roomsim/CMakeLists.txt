cmake_minimum_required(VERSION 3.18)

# CUDA 12 does not support GCC 13 as its host compiler. Respect an explicit
# compiler selection, but use GCC 12 where available as a compatible default.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(ROOMSIM_CUDA_HOST_COMPILER NAMES g++-12)
    if(ROOMSIM_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${ROOMSIM_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>)
target_link_libraries(roomsim PRIVATE m)
