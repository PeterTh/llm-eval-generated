cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present on the machine running the build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(nbody nbody.cpp)

# nbody.cpp contains both host code and CUDA kernels; compile it with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--extra-device-vectorization;-Xcompiler=-Wall,-Wextra,-march=native>
)

# Link math library and the CUDA runtime
target_link_libraries(nbody PRIVATE m CUDA::cudart)
