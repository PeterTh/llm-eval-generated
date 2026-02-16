cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# Force compiling the .cpp with the CUDA compiler so CUDA kernels are handled
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Optimization and warnings
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and CUDA runtime
find_package(CUDA REQUIRED)
target_link_libraries(qtclustering m CUDA::cudart)
