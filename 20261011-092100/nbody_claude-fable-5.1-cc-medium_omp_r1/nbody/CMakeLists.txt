cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations. FMA usage is written explicitly in the source
# (std::fma) so results are bit-reproducible; disable implicit contraction.
# -fno-math-errno lets sqrt vectorize (vsqrtpd); it does not alter results.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
