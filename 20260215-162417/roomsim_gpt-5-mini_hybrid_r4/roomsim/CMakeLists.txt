cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Treat the .cpp as CUDA source so nvcc compiles device kernels in this file
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options
target_compile_definitions(roomsim PRIVATE USE_MPI=1 USE_OMP=1 USE_CUDA=1)
if(OpenMP_CXX_FOUND)
    target_link_libraries(roomsim PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MPI_CXX_FOUND)
    target_link_libraries(roomsim PRIVATE MPI::MPI_CXX)
    target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

# Link CUDA runtime and math
target_link_libraries(roomsim PRIVATE CUDA::cudart m)

# Optimization flags
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>)
# NVCC flags
if(CMAKE_CUDA_COMPILER)
    target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)
    set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
