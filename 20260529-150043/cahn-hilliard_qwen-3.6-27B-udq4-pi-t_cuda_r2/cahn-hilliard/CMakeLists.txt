cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

enable_language(CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Force CUDA compilation for the .cpp source containing CUDA kernels
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "86"
)

target_compile_options(cahn_hilliard PRIVATE
    -O3 -Xcompiler -Wall -Xcompiler -Wextra
)

# Link math library and CUDA runtime
target_link_libraries(cahn_hilliard m)
