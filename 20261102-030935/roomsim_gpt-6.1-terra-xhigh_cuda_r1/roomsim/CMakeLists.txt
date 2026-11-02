cmake_minimum_required(VERSION 3.18)
# Supply native code for common GPUs and PTX for forward compatibility. The
# caller can override this with -DCMAKE_CUDA_ARCHITECTURES=... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(ROOMSIM_DEFAULT_CUDA_ARCHITECTURES ON)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)
if(ROOMSIM_DEFAULT_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
      "CUDA architectures for roomsim" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark keeps its original source file and executable name.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations while preserving the original arithmetic order
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native>)

# Link math library
target_link_libraries(roomsim m)
