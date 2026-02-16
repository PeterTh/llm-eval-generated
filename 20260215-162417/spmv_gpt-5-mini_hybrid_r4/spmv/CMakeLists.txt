cmake_minimum_required(VERSION 3.10)
# Set a default CUDA architecture to avoid detection failures when CUDA toolchain
set(CMAKE_CUDA_ARCHITECTURES 60)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)

add_executable(spmv spmv.cpp)

# If CUDA toolkit is found, enable CUDA language support and compile the source as CUDA
if (CUDAToolkit_FOUND)
    # Try to find an available CUDA compiler (nvcc or cudacc). If not found, fall back to CPU-only build.
    find_program(CUDACXX_EXECUTABLE NAMES nvcc cudacc PATHS ENV PATH NO_DEFAULT_PATH)
    if (CUDACXX_EXECUTABLE)
        message(STATUS "CUDAToolkit and CUDA compiler found, enabling CUDA path: ${CUDACXX_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER ${CUDACXX_EXECUTABLE} CACHE PATH "" FORCE)
        enable_language(CUDA)
        set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(spmv PRIVATE USE_CUDA=1)
        # prefer CUDA::cudart if available
        if (TARGET CUDA::cudart)
            target_link_libraries(spmv PRIVATE CUDA::cudart)
        elseif (TARGET CUDAToolkit::CUDART)
            target_link_libraries(spmv PRIVATE CUDAToolkit::CUDART)
        endif()
    else()
        message(STATUS "CUDAToolkit found but no CUDA compiler detected; building CPU-only fallback (USE_CUDA=0)")
        target_compile_definitions(spmv PRIVATE USE_CUDA=0)
    endif()
else()
    message(STATUS "CUDAToolkit not found, building CPU-only fallback (USE_CUDA=0)")
    target_compile_definitions(spmv PRIVATE USE_CUDA=0)
endif()

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
