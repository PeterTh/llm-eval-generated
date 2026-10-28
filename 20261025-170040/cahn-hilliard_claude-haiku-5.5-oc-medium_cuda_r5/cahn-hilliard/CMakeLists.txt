cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The source is CUDA: compile it with nvcc (no new files are introduced).
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# --fmad=false: no fused multiply-add contraction on the device, so every
# floating-point operation rounds exactly as written (reproducible results).
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-Wall,-Wextra,-march=native>
)

# Link math library (CUDA runtime is linked automatically by nvcc/CMake)
target_link_libraries(cahn_hilliard m)
