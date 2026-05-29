cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- find parallel runtimes ----
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

# Force CUDA compilation for the .cpp source (it contains CUDA kernels)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# ---- compile flags: O3, native, OpenMP, warnings ----
target_compile_options(cahn_hilliard PRIVATE
    -O3 -march=native -Wall -Wextra
    ${OpenMP_CXX_FLAGS})

# ---- link MPI, OpenMP, CUDA runtime, math ----
target_link_libraries(cahn_hilliard
    ${MPI_CXX_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    cudart
    m)

# ---- include directories ----
target_include_directories(cahn_hilliard PRIVATE
    ${MPI_CXX_INCLUDE_DIRS})

# ---- CUDA compilation flags ----
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
