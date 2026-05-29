cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# nvcc flags: -O3 for device, -Xcompiler for host (OpenMP + native opts)
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra
)

target_include_directories(black_scholes PRIVATE ${MPI_INCLUDE_PATH})

# Link MPI, OpenMP, and math
target_link_libraries(black_scholes
    ${MPI_CXX_LIBRARIES}
    -fopenmp
    m
)

target_compile_definitions(black_scholes PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
