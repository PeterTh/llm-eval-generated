#include <mpi.h>
#include <stdio.h>
#include <vector>

int main() {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) printf("Starting test with %d ranks\n", size);
    
    int nx = 4, ny = 4;
    size_t nz = 4;
    
    size_t nz_local = nz / size;
    size_t nz_remainder = nz % size;
    if (rank < static_cast<int>(nz_remainder)) {
        nz_local++;
    }
    
    printf("Rank %d: nz_local=%zu\n", rank, nz_local);
    
    size_t plane_size = nx * ny;
    std::vector<double> data((nz_local + 2) * plane_size, rank + 1.0);
    
    printf("Rank %d: allocated %zu elements, planes=%zu\n", rank, data.size(), (nz_local + 2));
    
    MPI_Finalize();
    return 0;
}
