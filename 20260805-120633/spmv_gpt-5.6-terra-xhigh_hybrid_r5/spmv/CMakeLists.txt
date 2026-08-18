cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

# The one executable deliberately always uses all three levels of parallelism:
# MPI distributes matrix rows between nodes, CUDA executes each local SpMV,
# and OpenMP parallelizes host-side CSR preparation and validation.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Avoid CMake's conservative compiler-default architecture (often sm_52).
# The small fat binary supports the NVIDIA accelerator generations commonly
# deployed in clusters while retaining architecture-specific machine code.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89)
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
else()
    set(CMAKE_CUDA_ARCHITECTURES 70 75)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(spmv spmv.cpp)

# Compile the benchmark itself with nvcc so its CUDA kernel is always present;
# there is intentionally no CPU-only executable or fallback path.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(spmv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    # nvcc compiles the host portions of spmv.cpp, so pass OpenMP explicitly
    # through to its host compiler; the imported OpenMP target supplies libgomp.
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m)
