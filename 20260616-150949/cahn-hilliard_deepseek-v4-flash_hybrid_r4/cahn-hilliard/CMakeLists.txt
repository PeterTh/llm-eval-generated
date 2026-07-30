cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Set the .cpp file to be compiled as CUDA
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (use -Xcompiler for nvcc to forward to host compiler)
target_compile_options(cahn_hilliard PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# Get MPI compile and link flags (nvcc doesn't inherit from MPI::MPI_CXX target)
execute_process(COMMAND mpicxx --showme:compile
                OUTPUT_VARIABLE MPI_CXX_FLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_CXX_FLAGS)
target_compile_options(cahn_hilliard PRIVATE ${MPI_CXX_FLAGS})

execute_process(COMMAND mpicxx --showme:link
                OUTPUT_VARIABLE MPI_LINK_FLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_LINK_FLAGS)

# Link libraries
target_link_libraries(cahn_hilliard PRIVATE
    ${MPI_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)

# Set CUDA architectures for modern GPUs (Ampere+)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90")
