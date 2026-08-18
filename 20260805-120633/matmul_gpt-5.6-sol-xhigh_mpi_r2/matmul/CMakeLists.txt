cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Use the MPI C API from C++ and avoid pulling in removed/deprecated MPI C++
# bindings on implementations that still expose them.
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
elseif(MSVC)
    target_compile_options(matmul PRIVATE /O2 /W4)
endif()

target_link_libraries(matmul PRIVATE MPI::MPI_CXX)
