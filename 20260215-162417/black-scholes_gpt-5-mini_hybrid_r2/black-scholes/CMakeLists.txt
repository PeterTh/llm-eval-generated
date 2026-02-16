cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations and hybrid parallel flags (MPI + OpenMP + OpenMP target offload)
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
# Ensure OpenMP host flags
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX)
else()
    target_compile_options(black_scholes PRIVATE -fopenmp)
    target_link_libraries(black_scholes PRIVATE -fopenmp)
endif()

# Add OpenMP device offload flags for NVIDIA targets if supported (best-effort)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp-targets=nvptx64-nvidia-cuda" COMPILER_SUPPORTS_OPENMP_OFFLOAD)
if(COMPILER_SUPPORTS_OPENMP_OFFLOAD)
    target_compile_options(black_scholes PRIVATE -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda)
    # try adding architecture flag if supported
    check_cxx_compiler_flag("-Xopenmp-target -march=sm_70" COMPILER_SUPPORTS_OPENMP_OFFLOAD_ARCH)
    if(COMPILER_SUPPORTS_OPENMP_OFFLOAD_ARCH)
        target_compile_options(black_scholes PRIVATE -Xopenmp-target -march=sm_70)
    endif()
else()
    # Fallback: ensure at least OpenMP host runtime is enabled
    target_compile_options(black_scholes PRIVATE -fopenmp)
endif()

# Link MPI and math library
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
