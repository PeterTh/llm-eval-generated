#include <mpi.h>
#include <stdio.h>
#include <vector>

// Test ghost exchange with empty ranks
inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Test case: nz=3, size=5
    int nz = 3;
    int nx = 2, ny = 2;
    int base_local_nz = nz / size;
    int remainder = nz % size;
    int local_nz = base_local_nz + (rank < remainder ? 1 : 0);
    
    size_t plane_size = nx * ny;
    size_t buffer_size = (local_nz + 2) * plane_size;
    std::vector<double> field(buffer_size, 0.0);
    
    // Initialize real data with rank number
    for (int z = 1; z <= local_nz; ++z) {
        for (size_t i = 0; i < plane_size; ++i) {
            field[z * plane_size + i] = rank * 10.0 + z;
        }
    }
    
    printf("Rank %d (local_nz=%d): Before exchange - plane[0]=%.1f plane[1]=%.1f plane[%d]=%.1f\n",
           rank, local_nz, 
           field[0], 
           field[plane_size], 
           local_nz+1,
           field[(local_nz+1) * plane_size]);
    
    // Ghost exchange
    int top_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int bottom_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    
    // CRITICAL: When local_nz=0:
    // - idx3(0,0,local_nz) = idx3(0,0,0) = 0 (bottom ghost!)
    // - idx3(0,0,1) = plane_size (top ghost!)
    // This means rank sends GHOST DATA, not real data!
    
    MPI_Sendrecv(
        &field[idx3(0, 0, local_nz, nx, ny)], plane_size, MPI_DOUBLE, top_neighbor, 0,
        &field[idx3(0, 0, 0, nx, ny)],        plane_size, MPI_DOUBLE, bottom_neighbor, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    
    MPI_Sendrecv(
        &field[idx3(0, 0, 1, nx, ny)],          plane_size, MPI_DOUBLE, bottom_neighbor, 1,
        &field[idx3(0, 0, local_nz + 1, nx, ny)], plane_size, MPI_DOUBLE, top_neighbor, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    
    printf("Rank %d: After exchange - plane[0]=%.1f plane[1]=%.1f plane[%d]=%.1f\n",
           rank,
           field[0], 
           field[plane_size], 
           (local_nz+1),
           field[(local_nz+1) * plane_size]);
    
    MPI_Finalize();
    return 0;
}
