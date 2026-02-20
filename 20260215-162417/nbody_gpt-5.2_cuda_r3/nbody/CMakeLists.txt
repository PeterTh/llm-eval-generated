cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xptxas=-O3>
)

# CUDA architectures (use modern CMake setting when available; otherwise pass NVCC flags)
if (CMAKE_VERSION VERSION_LESS 3.18)
  target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -gencode=arch=compute_70,code=sm_70
      -gencode=arch=compute_75,code=sm_75
      -gencode=arch=compute_80,code=sm_80
      -gencode=arch=compute_86,code=sm_86
      -gencode=arch=compute_70,code=compute_70>
  )
else()
  set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES 70;75;80;86)
endif()

# Link math library
target_link_libraries(nbody m)
