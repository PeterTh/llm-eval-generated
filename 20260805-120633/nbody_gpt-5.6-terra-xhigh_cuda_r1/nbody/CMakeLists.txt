cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nbody.cpp contains both the small host-side command-line wrapper and the
# CUDA kernels.  Keeping the source name preserves the executable/source
# layout while compiling the complete simulation with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Compile for the GPU present on the build machine.  The benchmark therefore
# always uses the native CUDA architecture instead of a CPU fallback.
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

# Retained for the host-side validation and result reporting code.
target_link_libraries(nbody m)
