cmake_minimum_required(VERSION 3.24)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile for the GPU(s) present on the build/run machine.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# cholesky.cpp contains CUDA kernels; compile it with nvcc while keeping MPI,
# OpenMP and the existing executable name.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES LINKER_LANGUAGE CUDA)

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(cholesky PRIVATE m ${MPI_CXX_LIBRARIES} gomp pthread)
