cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable from single source (compiled as CUDA source to enable device kernels in the same file)
add_executable(black_scholes black_scholes.cpp)

# Treat the .cpp as CUDA source so NVCC compiles device code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable high optimizations and standards for host and device
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fPIC>
)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Link libraries
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Ensure CUDA uses device runtime linking properly
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add flags for position independent code if needed when linking against MPI
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(black_scholes PRIVATE -fPIC)
endif()
