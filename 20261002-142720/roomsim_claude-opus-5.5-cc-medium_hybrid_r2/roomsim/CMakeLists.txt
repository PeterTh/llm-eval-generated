cmake_minimum_required(VERSION 3.18)

# GPU architectures: the GPUs of the build machine if available, otherwise common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  execute_process(COMMAND nvidia-smi -L RESULT_VARIABLE _smi_result OUTPUT_QUIET ERROR_QUIET)
  if(_smi_result EQUAL 0 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
  endif()
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Hybrid MPI + OpenMP + CUDA: the single source file is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings. Device code is compiled without automatic FMA
# contraction; the kernels spell out the host compiler's contraction pattern explicitly
# so that results are bitwise identical to the host reference.
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --extended-lambda -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)

# MPI headers/libraries are added directly (MPI compile options such as -pthread are not
# understood by nvcc)
target_include_directories(roomsim SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
