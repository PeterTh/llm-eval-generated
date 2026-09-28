cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -ffp-contract=off keeps floating-point rounding independent of vectorization;
# the kernels spell out the fused multiply-adds explicitly with std::fma(), so
# results stay bit-identical to the original sequential code.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(nbody m OpenMP::OpenMP_CXX)
