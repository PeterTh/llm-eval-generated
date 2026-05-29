cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Build as a CUDA-hosted executable (nvcc compiles both host C++ and device code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture
execute_process(
    COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .\n'"
    OUTPUT_VARIABLE DETECTED_ARCH
    ERROR_QUIET
)
if(NOT DETECTED_ARCH OR DETECTED_ARCH STREQUAL "")
    set(DETECTED_ARCH "80")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "CUDA architectures" FORCE)

add_executable(floydwarshall floydwarshall.cpp)

# Force nvcc compilation so CUDA kernels are compiled
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    --expt-relaxed-constexpr
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link MPI, OpenMP, and math
target_link_libraries(floydwarshall
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Include the common directory for shared headers
target_include_directories(floydwarshall PRIVATE ${CMAKE_SOURCE_DIR}/../common)
