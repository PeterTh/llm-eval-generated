cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# The benchmark always uses shared-memory OpenMP parallelism; fail configuration
# rather than producing a serial executable when OpenMP is unavailable.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and OpenMP runtimes.  The imported OpenMP target supplies the
# compiler flags needed for both the OpenMP pragmas and runtime.
target_link_libraries(black_scholes PRIVATE m OpenMP::OpenMP_CXX)
