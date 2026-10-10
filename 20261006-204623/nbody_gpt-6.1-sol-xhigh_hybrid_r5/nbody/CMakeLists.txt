cmake_minimum_required(VERSION 3.24)

# Optimize for the build node's GPUs. Cluster/cross builds may override this
# (for example, -DCMAKE_CUDA_ARCHITECTURES=80).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile CUDA in the existing file, keeping the executable name.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Keep precise double arithmetic and the original force summation order.
target_compile_options(nbody PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-O3,-Wall,-Wextra>"
)
separate_arguments(nbody_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS nbody_openmp_flags)
    target_compile_options(nbody PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Detect CUDA support in Open MPI and MPICH-derived implementations. MPI
# implementations without a GPU-support query use portable pinned host staging.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
    #include <mpi.h>
    #include <mpi-ext.h>
    int main() { return MPIX_Query_cuda_support(); }
" NBODY_MPI_CUDA_QUERY)
if(NOT NBODY_MPI_CUDA_QUERY)
    check_cxx_source_compiles("
        #include <mpi.h>
        int main() {
            int supported = 0;
            return MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA, &supported);
        }
    " NBODY_MPI_GPU_QUERY)
endif()
unset(CMAKE_REQUIRED_LIBRARIES)
if(NBODY_MPI_CUDA_QUERY)
    target_compile_definitions(nbody PRIVATE NBODY_MPI_CUDA_QUERY=1)
elseif(NBODY_MPI_GPU_QUERY)
    target_compile_definitions(nbody PRIVATE NBODY_MPI_GPU_QUERY=1)
endif()
