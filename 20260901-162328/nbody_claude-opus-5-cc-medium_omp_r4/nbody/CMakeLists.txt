cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -ffp-contract=off pins floating-point rounding: every FMA the reference
# implementation performs is written out explicitly with std::fma in the
# kernels, so the parallel version stays bit-identical regardless of how the
# compiler chooses to vectorize. -fno-math-errno does not change any result,
# it only lets sqrt/fma be emitted as plain instructions so the SIMD kernel
# actually vectorizes.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(nbody m OpenMP::OpenMP_CXX)
