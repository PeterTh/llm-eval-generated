cmake_minimum_required(VERSION 3.18)

# Ensure nvcc is found even if PATH isn't configured.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile the existing .cpp with nvcc so we can embed CUDA kernels without adding new files.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --fmad=false --prec-div=true --prec-sqrt=true -Xcompiler=-march=native,-Wall,-Wextra -Xcompiler=-fopenmp>
)

target_link_libraries(qtclustering PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
