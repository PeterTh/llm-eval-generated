cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Force the .cpp file through nvcc for CUDA kernel support
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Set include directories
target_include_directories(black_scholes PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../common
)

# Enable optimizations for CUDA
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "86"  # RTX 3090 = Ampere SM 86
)
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

# Link libraries
target_link_libraries(black_scholes
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cuda
    cudart
    m
)
