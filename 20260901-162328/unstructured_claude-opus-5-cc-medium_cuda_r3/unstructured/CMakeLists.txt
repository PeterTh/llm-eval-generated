cmake_minimum_required(VERSION 3.18)

# nvcc only accepts host compilers up to a version-dependent maximum; fall back
# to an older GCC when the default one is too new for the installed toolkit.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(NVCC_EXECUTABLE nvcc)
  if(NVCC_EXECUTABLE)
    execute_process(COMMAND ${NVCC_EXECUTABLE} --version OUTPUT_VARIABLE NVCC_VERSION_TEXT)
    string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _m "${NVCC_VERSION_TEXT}")
    set(NVCC_MAJOR ${CMAKE_MATCH_1})
    set(NVCC_MINOR ${CMAKE_MATCH_2})
    if(NVCC_MAJOR VERSION_LESS 12 OR (NVCC_MAJOR VERSION_EQUAL 12 AND NVCC_MINOR VERSION_LESS 4))
      set(MAX_GNU_MAJOR 12)
    else()
      set(MAX_GNU_MAJOR 13)
    endif()
    execute_process(COMMAND g++ -dumpfullversion -dumpversion
                    OUTPUT_VARIABLE HOST_GXX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    string(REGEX MATCH "^([0-9]+)" _m2 "${HOST_GXX_VERSION}")
    if(CMAKE_MATCH_1 AND CMAKE_MATCH_1 VERSION_GREATER MAX_GNU_MAJOR)
      find_program(FALLBACK_GXX g++-${MAX_GNU_MAJOR})
      if(FALLBACK_GXX)
        set(CMAKE_CUDA_HOST_COMPILER ${FALLBACK_GXX})
      endif()
    endif()
  endif()
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89)
endif()

# The benchmark source is CUDA C++ even though it keeps its .cpp extension
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations. -fmad=false keeps the floating-point expressions
# unfused, so the GPU reproduces the reference results bit for bit.
target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(unstructured m)
