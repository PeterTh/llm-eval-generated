cmake_minimum_required(VERSION 3.24)

# Target the installed accelerators by default; cross builds can supply a list
# through -DCMAKE_CUDA_ARCHITECTURES (including PTX for newer cluster nodes).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA without introducing another source file.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
target_compile_definitions(black_scholes PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Retain IEEE double precision and the formula's operation order on the GPU.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(black_scholes PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-ffp-contract=off>
    )
endif()

# OpenMP's imported target supplies its runtime, but its compile flags are
# normally restricted to CXX and must also reach nvcc's host compiler.
separate_arguments(BLACK_SCHOLES_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS BLACK_SCHOLES_OPENMP_FLAGS)
    target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
