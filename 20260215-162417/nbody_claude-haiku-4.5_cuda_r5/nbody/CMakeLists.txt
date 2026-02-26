add_executable(nbody nbody.cpp)

# Mark the file as CUDA code
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

# Link math library and CUDA runtime
target_link_libraries(nbody m)
