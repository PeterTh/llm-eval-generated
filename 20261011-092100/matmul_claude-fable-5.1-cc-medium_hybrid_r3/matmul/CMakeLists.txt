cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single source file contains CUDA kernels alongside MPI/OpenMP host code,
# so it is compiled with the CUDA compiler.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# Target the GPUs of the build machine when CMake supports it, otherwise a set
# of common FP64-capable architectures.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES native)
else()
  set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "70;80;86;90")
endif()
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(matmul PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})

# Host-side optimisation / OpenMP flags are forwarded through nvcc.
target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra,${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffp-contract=fast -Wall -Wextra ${OpenMP_CXX_FLAGS}>
)

target_link_libraries(matmul PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
target_link_options(matmul PRIVATE ${MPI_CXX_LINK_FLAGS})
