cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 CACHE STRING "CUDA architectures" FORCE)

# Use nvcc to compile the source (contains embedded CUDA kernels)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xcompiler="-fopenmp,-O3,-march=native,-Wall,-Wextra")

# Build the executable
add_executable(black_scholes black_scholes.cpp)

# Link MPI, OpenMP, and CUDA runtime
target_link_libraries(black_scholes MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_cudart_LIBRARY} m)

# Include directories
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
