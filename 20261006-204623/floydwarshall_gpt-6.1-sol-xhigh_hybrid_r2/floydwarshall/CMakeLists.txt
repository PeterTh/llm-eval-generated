cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source and executable names; nvcc compiles the CUDA kernels.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)

target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>)
# OpenMP's imported target supplies host linkage. Forward its compile flags to
# nvcc as well, since the existing .cpp file is compiled as CUDA.
separate_arguments(FLOYD_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS FLOYD_OPENMP_FLAGS)
    target_compile_options(floydwarshall PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(floydwarshall PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
