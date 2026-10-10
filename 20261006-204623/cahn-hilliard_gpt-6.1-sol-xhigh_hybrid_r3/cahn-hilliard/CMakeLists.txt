cmake_minimum_required(VERSION 3.20)

# Override with -DCMAKE_CUDA_ARCHITECTURES=... for the cluster's accelerators.
# Include native code for common GPUs and PTX for newer architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Detect the MPI implementation's runtime CUDA-support query. This keeps
# ordinary MPI builds portable and enables device-buffer communication when
# supported, without a mandatory vendor-specific header or dependency.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("#include <mpi.h>\nint main() { return MPIX_Query_cuda_support(); }"
                         CH_MPI_CUDA_QUERY)
if(NOT CH_MPI_CUDA_QUERY)
    check_cxx_source_compiles("#include <mpi.h>\n#include <mpi-ext.h>\nint main() { return MPIX_Query_cuda_support(); }"
                             CH_MPI_CUDA_QUERY_EXT)
endif()
unset(CMAKE_REQUIRED_LIBRARIES)

# Compile the existing file as CUDA without renaming it or adding sources.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
if(CH_MPI_CUDA_QUERY)
    target_compile_definitions(cahn_hilliard PRIVATE CH_MPI_CUDA_QUERY=1)
elseif(CH_MPI_CUDA_QUERY_EXT)
    target_compile_definitions(cahn_hilliard PRIVATE CH_MPI_CUDA_QUERY=1 CH_MPI_CUDA_QUERY_EXT=1)
endif()

# Keep the original double-precision expression order; do not use fast math.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>)
separate_arguments(CH_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS CH_OPENMP_FLAGS)
    target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()
