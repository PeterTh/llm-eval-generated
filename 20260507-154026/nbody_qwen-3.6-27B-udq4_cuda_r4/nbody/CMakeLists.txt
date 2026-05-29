cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations – target the detected GPU architecture
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "86"
)
target_compile_options(nbody PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(nbody m)
