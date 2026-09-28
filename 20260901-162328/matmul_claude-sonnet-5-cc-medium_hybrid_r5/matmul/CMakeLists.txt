cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The benchmark source combines MPI, OpenMP and CUDA (kernel launches) in a
# single .cpp file. Generate a .cu translation unit from it at build time so
# nvcc compiles the CUDA parts, without renaming the tracked source file.
set(MATMUL_CU ${CMAKE_CURRENT_BINARY_DIR}/matmul_generated.cu)
add_custom_command(
    OUTPUT ${MATMUL_CU}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp ${MATMUL_CU}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    COMMENT "Generating CUDA translation unit from matmul.cpp"
    VERBATIM
)

add_executable(matmul ${MATMUL_CU})

# Enable optimizations. OpenMP/architecture flags are forwarded to the host
# compiler via -Xcompiler since nvcc drives the compilation of this file.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_link_options(matmul PRIVATE -fopenmp)
