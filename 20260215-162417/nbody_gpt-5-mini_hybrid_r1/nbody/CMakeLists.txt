cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(SRC nbody.cpp)
# Force compilation of the C++ source file with the CUDA toolchain so device kernels can be in the same file.
set_source_files_properties(${SRC} PROPERTIES LANGUAGE CUDA)

add_executable(nbody ${SRC})

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
# Enable CUDA fast math and aggressive optimization for device code
if (CUDA_TOOLKIT_ROOT_DIR)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)
endif()

# Link MPI and OpenMP and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Expose parallelization defines to the source
target_compile_definitions(nbody PRIVATE USE_MPI=1 USE_OPENMP=1 USE_CUDA=1)
