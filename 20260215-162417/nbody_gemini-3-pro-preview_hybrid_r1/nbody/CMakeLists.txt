cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_60,code=sm_60 -O3 -Xcompiler -O3,-march=native,-Wall,-Wextra>)

# Link libraries
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
