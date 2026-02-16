cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Optional CUDA: enable if a CUDA compiler is available, otherwise compile CPU-only fallback
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAVE_CUDA ON)
else()
    set(HAVE_CUDA OFF)
endif()

# Create executable
add_executable(roomsim roomsim.cpp)

if(HAVE_CUDA)
    # Treat the single source as CUDA to allow device kernels in the same file
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>)
    set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(roomsim PRIVATE USE_CUDA=1)
else()
    message(WARNING "CUDA compiler not found; building CPU-only CUDA fallback (OpenMP+MPI still enabled)")
    target_compile_definitions(roomsim PRIVATE USE_CUDA=0)
endif()

# Compiler flags
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)

# Link libraries: math, MPI, OpenMP
target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
