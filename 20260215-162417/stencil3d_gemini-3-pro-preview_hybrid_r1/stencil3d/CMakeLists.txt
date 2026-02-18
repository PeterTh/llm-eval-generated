cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations and set architecture
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Set CUDA Architectures (Ampere 86 for RTX 3090, but include others for portability)
    set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES "native")

    target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>) 
    target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
else()
    target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# If we are compiling .cpp as CUDA, we need to tell CMake
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
