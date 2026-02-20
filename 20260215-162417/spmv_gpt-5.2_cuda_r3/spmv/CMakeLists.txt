cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Compile the existing .cpp with NVCC so we can include CUDA kernels without adding new files.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# Note: COMPILE_LANGUAGE:CXX flags won't apply to spmv.cpp after LANGUAGE CUDA is set,
# so pass host flags via -Xcompiler for NVCC.
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xcompiler=-O3 -Xcompiler=-march=native>
)

# Prefer building for the local GPU when supported by the CMake version/toolchain.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES native)
endif()

# Link math library; CUDA runtime is linked implicitly when using the CUDA language.
target_link_libraries(spmv PRIVATE m)
