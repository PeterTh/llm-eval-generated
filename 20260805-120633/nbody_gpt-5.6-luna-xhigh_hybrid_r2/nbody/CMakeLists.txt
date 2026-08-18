cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build a native cubin for the RTX 30/Ampere accelerator class used by the
# benchmark environment. Cluster builds can override this cache variable with
# the installed GPU architecture(s), e.g. -DNBODY_CUDA_ARCHITECTURES=90.
set(NBODY_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA compute capability for the target accelerator")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the required executable name while compiling the existing translation
# unit with nvcc so the MPI/OpenMP host code and CUDA kernel are built together.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES "${NBODY_CUDA_ARCHITECTURES}")

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

target_link_libraries(nbody PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
