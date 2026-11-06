cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(NBODY_DEFAULT_CUDA_ARCHITECTURES ON)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Cover common cluster GPUs with native code and retain PTX for newer GPUs.
# A caller can override this with -DCMAKE_CUDA_ARCHITECTURES for its cluster.
if(NBODY_DEFAULT_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "52-real;60-real;70-real;80" CACHE STRING "CUDA architectures" FORCE)
endif()
# Keep the existing source and executable names while compiling the kernels with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
