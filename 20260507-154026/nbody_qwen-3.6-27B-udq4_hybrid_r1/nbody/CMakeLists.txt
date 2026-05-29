cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find MPI, OpenMP, CUDA ---------------------------------------
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# ---- Build the hybrid executable (host = C++, device = CUDA) ------
# Compile the single source with nvcc so the __global__ kernel and
# runtime calls are handled correctly.
add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CUDA)

# ---- Optimizations ------------------------------------------------
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Pass -fopenmp to the host compiler via nvcc so OpenMP pragmas work
set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler;-fopenmp>)

# ---- Link libraries -----------------------------------------------
target_link_libraries(nbody
    ${MPI_CXX_LIBRARIES}
    gomp
    m
)

# ---- Compile definitions (MPI include dirs + project includes) ----
target_include_directories(nbody PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/..
)
