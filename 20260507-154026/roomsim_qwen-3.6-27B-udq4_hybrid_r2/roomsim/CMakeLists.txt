cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

add_executable(roomsim roomsim.cpp)

# Force nvcc compilation for the CUDA source
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -Xcompiler=-fopenmp,-march=native,-Wall,-Wextra)

# Link MPI, OpenMP, CUDA runtime, and math library
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(roomsim ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gomp cudart m)
