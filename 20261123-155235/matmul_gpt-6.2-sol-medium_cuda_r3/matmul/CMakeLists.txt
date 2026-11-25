cmake_minimum_required(VERSION 3.20)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and compile it as CUDA.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-O3,-Wall,-Wextra)

# The original GCC x86 build used separate multiply/add for vectorized pairs
# and contracted the final scalar iteration when the host supported FMA.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_cxx_source_compiles("
        #ifndef __FMA__
        #error FMA is unavailable
        #endif
        int main() {}
    " MATMUL_FUSED_TAIL)
    unset(CMAKE_REQUIRED_FLAGS)
    if(MATMUL_FUSED_TAIL)
        target_compile_definitions(matmul PRIVATE MATMUL_FUSED_TAIL)
    endif()
endif()
