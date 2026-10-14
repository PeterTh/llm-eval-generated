cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on the build machine
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The source is compiled as CUDA (nvcc); host-side code is still built with the
# host compiler flags below
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; disable FMA contraction on the device so results match
# strict IEEE evaluation order
target_compile_options(cahn_hilliard PRIVATE
    -O3 --fmad=false -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra
)

# Link math library and CUDA runtime
target_link_libraries(cahn_hilliard m CUDA::cudart)
