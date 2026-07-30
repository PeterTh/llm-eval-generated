cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 30xx series (Ampere)
endif()

# Find MPI
find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Force cholesky.cpp to be compiled with nvcc (CUDA compiler)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Compile options: optimizations + CUDA + OpenMP + MPI
target_compile_options(cholesky
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 -Xcompiler=-fopenmp,-march=native,-Wall,-Wextra
            -Xcompiler=${MPI_CXX_COMPILE_FLAGS}
        >
)

# Define HAVE_CUDA so code can conditionally compile CUDA paths
target_compile_definitions(cholesky PRIVATE HAVE_CUDA)

# Link libraries: MPI, OpenMP, CUDA runtime, math
target_link_libraries(cholesky
    PRIVATE
        ${MPI_CXX_LIBRARIES}
        ${MPI_CXX_LINK_FLAGS}
        -fopenmp
        m
)

# Include directories
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})
