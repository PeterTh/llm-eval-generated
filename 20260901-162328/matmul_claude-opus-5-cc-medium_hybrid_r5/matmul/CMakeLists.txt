cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 does not support GCC >= 13 as host compiler; fall back if needed.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(HOST_GXX NAMES g++)
  if(HOST_GXX)
    execute_process(COMMAND ${HOST_GXX} -dumpfullversion -dumpversion
                    OUTPUT_VARIABLE HOST_GXX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HOST_GXX_VERSION VERSION_GREATER_EQUAL 13)
      find_program(GXX12 NAMES g++-12 g++-11)
      if(GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12})
      endif()
    endif()
  endif()
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The single translation unit contains host (MPI/OpenMP) and device code
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Host-side optimizations are forwarded to the host compiler by nvcc
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
