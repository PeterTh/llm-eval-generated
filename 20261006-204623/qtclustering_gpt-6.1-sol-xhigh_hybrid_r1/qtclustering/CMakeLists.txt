cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

include(CheckCXXSourceCompiles)
set(qt_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -march=native")
check_cxx_source_compiles("
#if !defined(__FMA__) && !defined(__ARM_FEATURE_FMA)
#error No native fused multiply-add
#endif
int main() { return 0; }
" QT_NATIVE_FMA)
set(CMAKE_REQUIRED_FLAGS "${qt_saved_required_flags}")

# Keep the original source and executable names; nvcc compiles the kernels
# embedded in the existing .cpp file. Override architectures when cross-building.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)
target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
if(QT_NATIVE_FMA)
    target_compile_definitions(qtclustering PRIVATE QT_NATIVE_FMA=1)
endif()

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
