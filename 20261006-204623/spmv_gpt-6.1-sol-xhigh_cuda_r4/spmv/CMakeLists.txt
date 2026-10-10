cmake_minimum_required(VERSION 3.24)

# Use the installed GPU by default; callers can still select architectures with
# -DCMAKE_CUDA_ARCHITECTURES when building for a different machine.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(spmv_benchmark LANGUAGES CXX CUDA)

# Keep the existing source and executable names, compiling the source with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# CMake links the CUDA runtime automatically. No additional libraries are needed.
target_link_libraries(spmv m)
