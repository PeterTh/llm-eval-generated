cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark source contains CUDA code; compile it with nvcc while keeping
# the original file name (and therefore the executable name) unchanged.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

# Compile for the GPU(s) present on the build machine so no PTX JIT is needed
# at run time (CMake otherwise defaults to an old architecture such as sm_52).
if(CMAKE_VERSION VERSION_LESS 3.24)
    set(_matmul_cuda_arch 86)
else()
    set(_matmul_cuda_arch native)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${_matmul_cuda_arch} CACHE STRING "CUDA GPU architectures" FORCE)

add_executable(matmul matmul.cpp)

# Enable optimizations for both the host and device compilation passes.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Wall;-Wextra>)

# Link math library and the CUDA runtime.
target_link_libraries(matmul PRIVATE m CUDA::cudart)
