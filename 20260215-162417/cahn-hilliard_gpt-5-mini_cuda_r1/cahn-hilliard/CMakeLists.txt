cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the existing source with the CUDA compiler
add_executable(cahn_hilliard cahn_hilliard.cpp)
# Tell CMake to treat the .cpp as CUDA source so NVCC compiles it
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# Apply flags for both host and device compilation
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math and CUDA runtime libraries
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    target_link_libraries(cahn_hilliard PRIVATE m CUDA::cudart)
else()
    target_link_libraries(cahn_hilliard PRIVATE m)
endif()
