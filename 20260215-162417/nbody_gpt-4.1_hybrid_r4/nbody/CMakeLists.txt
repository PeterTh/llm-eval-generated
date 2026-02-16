cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
enable_language(CUDA)

add_executable(nbody nbody.cu)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATON ON)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(nbody PRIVATE /usr/local/cuda/include)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra $<$<BOOL:${OpenMP_FOUND}>:${OpenMP_CXX_FLAGS}>)

# Link math, MPI, OpenMP, CUDA
set(CMAKE_CUDA_STANDARD 11)
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
target_compile_features(nbody PRIVATE cuda_std_11)
