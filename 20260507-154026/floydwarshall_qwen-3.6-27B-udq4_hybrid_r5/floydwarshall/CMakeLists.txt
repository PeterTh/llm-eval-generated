cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find MPI ---
find_package(MPI REQUIRED)

# --- Find OpenMP ---
find_package(OpenMP REQUIRED)

# --- Enable CUDA ---
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Wrap the .cpp source so nvcc handles the CUDA kernel inside it
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# --- Optimizations ---
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# --- Link MPI + OpenMP + CUDA runtime + math ---
target_link_libraries(floydwarshall MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)

# Set include path for common headers
target_include_directories(floydwarshall PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
