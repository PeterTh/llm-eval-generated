cmake_minimum_required(VERSION 3.10)

# This benchmark unconditionally uses CUDA; prefer clang++ as the CUDA compiler if nvcc is absent.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_cuda_nvcc nvcc)
  if(_cuda_nvcc)
    set(CMAKE_CUDA_COMPILER "${_cuda_nvcc}" CACHE FILEPATH "" FORCE)
  else()
    find_program(_cuda_clang clang++)
    if(_cuda_clang)
      set(CMAKE_CUDA_COMPILER "${_cuda_clang}" CACHE FILEPATH "" FORCE)
    endif()
  endif()
endif()

# Default to Ampere (RTX 3090) for maximum performance in this environment.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "" FORCE)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(qtclustering m)
