cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)

add_executable(qtclustering qtclustering.cpp)

# Tell CMake to compile the .cpp as CUDA source so kernels in the same file are compiled
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Compiler flags
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-fopenmp>
)

# Define that we are using MPI/OpenMP/CUDA
target_compile_definitions(qtclustering PRIVATE USE_MPI USE_OPENMP USE_CUDA)

# Link libraries
target_link_libraries(qtclustering PRIVATE m MPI::MPI_C OpenMP::OpenMP_CXX)

# Enable CUDA separable compilation for better optimization
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
