cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)

# The trailing matrix update is the dominant kernel.  Native vector
# instructions and relaxed floating-point reassociation let the compiler emit
# packed multiply-add reductions while retaining ample accuracy for Cholesky.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(cholesky PRIVATE
        -O3 -march=native -ffast-math -Wall -Wextra
    )
endif()
