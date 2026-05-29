cmake_minimum_required(VERSION 3.17)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_library(CUBLAS_LIB cublas HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/lib/x86_64-linux-gnu")
find_library(CUSOLVER_LIB cusolver HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/lib/x86_64-linux-gnu")
find_library(CUSOLVER_LAPACK_LIB cusolver_lapack_static HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/lib/x86_64-linux-gnu")

add_executable(cholesky cholesky.cu)

target_compile_options(cholesky PRIVATE
    -O3 -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fopenmp>
)

target_link_libraries(cholesky
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CUBLAS_LIB}
    ${CUSOLVER_LIB}
    ${CUSOLVER_LAPACK_LIB}
    ${CUDA_cudart_LIBRARY}
    m
)

set_target_properties(cholesky PROPERTIES LINKER_LANGUAGE CXX)
