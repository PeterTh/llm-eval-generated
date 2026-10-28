cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on the machine unless the user overrides it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# The program is CUDA-parallel unconditionally: compile the source with nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings (host code is forwarded through nvcc -Xcompiler).
# -Wpedantic is not forwarded for CUDA: nvcc's preprocessed #line directives trigger
# GNU-extension warnings in every included header.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall,-Wextra>
)

# Link math library and CUDA runtime
target_link_libraries(roomsim PRIVATE m CUDA::cudart)
