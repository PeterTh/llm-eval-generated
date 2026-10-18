cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

# Compile the existing source with nvcc, forwarding OpenMP to its host compiler.
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
)
separate_arguments(OpenMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_HOST_FLAGS)
    target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
    CUDA::cudart CUDA::cublas CUDA::cusolver)
