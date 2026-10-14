cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU present on this machine unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

# The benchmark is compiled as CUDA so the simulation kernel lives in the same file
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    # Disable fused multiply-add contraction so device arithmetic follows the same
    # rounding sequence as the reference expression order
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler;-march=native;-Xcompiler;-Wall;-Xcompiler;-Wextra>
)

# Link math library and CUDA runtime
target_link_libraries(unstructured m CUDA::cudart)
