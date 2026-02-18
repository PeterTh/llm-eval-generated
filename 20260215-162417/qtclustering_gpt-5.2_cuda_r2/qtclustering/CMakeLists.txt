cmake_minimum_required(VERSION 3.18)

# Set architectures *before* enabling CUDA to avoid default-arch probing.
set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "" FORCE)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-O3,-march=native>
)

# Link math library + CUDA runtime
target_link_libraries(qtclustering PRIVATE m CUDA::cudart)
