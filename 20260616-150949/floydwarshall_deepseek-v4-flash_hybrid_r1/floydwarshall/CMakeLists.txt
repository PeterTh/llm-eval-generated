cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(floydwarshall floydwarshall.cpp)

# Mark source as CUDA so nvcc is used for compilation
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX)
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Find OpenMP
find_package(OpenMP REQUIRED)

# CUDA compile options: pass OpenMP and optimization flags through to host compiler
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra
        -lineinfo
        --restrict
        --use_fast_math
    >
)

# Link libraries: MPI, OpenMP, CUDA runtime, math
target_link_libraries(floydwarshall PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    cudart
    m
)
