cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
# Compile the .cpp with NVCC so CUDA kernels in the same file are compiled
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Optimization flags
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra> 
                                            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Link MPI, OpenMP and math
target_link_libraries(floydwarshall PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Ensure NVCC uses the host compiler's include paths
if(CMAKE_CXX_COMPILER)
  set_target_properties(floydwarshall PROPERTIES CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
