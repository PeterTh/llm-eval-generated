cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit QUIET)

add_executable(cholesky cholesky.cpp)

# If CUDA toolkit is found, compile cholesky.cpp with CUDA language (nvcc) and enable USE_CUDA
if (CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Toolkit and compiler found; enabling CUDA acceleration")
    # Provide a set of common architectures to satisfy CMake's requirement
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
    enable_language(CUDA)
    target_compile_definitions(cholesky PRIVATE USE_CUDA)
    set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>)
else()
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit found but no CUDA compiler (nvcc) available; building CPU-only fallback")
    else()
        message(STATUS "CUDA Toolkit not found; building CPU-only fallback")
    endif()
endif()

# Compile options for host
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# Link required libraries
target_link_libraries(cholesky PRIVATE MPI::MPI_C MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# If OpenMP flags exist, propagate to nvcc-hosted compilation as well
if(OpenMP_CXX_FOUND)
    if (CUDAToolkit_FOUND)
        target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;${OpenMP_CXX_FLAGS}>)
    endif()
endif()
