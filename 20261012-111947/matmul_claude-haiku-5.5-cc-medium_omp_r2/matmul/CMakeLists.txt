cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# OpenMP is required unconditionally
find_package(OpenMP REQUIRED)

# Enable optimizations
# -ffp-contract=off keeps mul+add unfused so results do not depend on how the
# compiler vectorizes/contracts the k-reduction (same bits at every thread count)
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(matmul m OpenMP::OpenMP_CXX)
