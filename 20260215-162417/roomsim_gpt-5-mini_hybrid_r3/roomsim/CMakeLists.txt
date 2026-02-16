cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)

add_executable(roomsim roomsim.cpp)
# Tell CMake to compile roomsim.cpp with the CUDA compiler so the small CUDA kernel is handled
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>)
# CUDA-specific flags
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link MPI and OpenMP and math; rely on imported targets
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cudart)

# Ensure CUDA runtime linkage when using NVCC
if (CMAKE_CUDA_COMPILER)
    set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
