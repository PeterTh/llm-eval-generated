#include <mpi.h>
#include <stdio.h>
#include <vector>

// Simulate the key issue: what happens with local_nz = 0?
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Test case: nz=3, size=5
    int nz = 3;
    int base_local_nz = nz / size;  // 0
    int remainder = nz % size;       // 3
    
    // Ranks 0,1,2 get 1 plane each
    // Ranks 3,4 get 0 planes
    int local_nz = base_local_nz + (rank < remainder ? 1 : 0);
    
    printf("Rank %d: local_nz = %d\n", rank, local_nz);
    
    // Test the loop behavior
    int loop_count = 0;
    for (size_t z = 1; z <= (size_t)local_nz; ++z) {
        loop_count++;
    }
    printf("Rank %d: loop executed %d times\n", rank, loop_count);
    
    // Test buffer allocation
    int nx = 4, ny = 4;
    size_t plane_size = nx * ny;
    size_t local_buffer_size = (local_nz + 2) * plane_size;
    
    printf("Rank %d: buffer_size = %zu (should have %d planes + 2 ghosts)\n", 
           rank, local_buffer_size, local_nz);
    
    std::vector<double> buffer(local_buffer_size, rank * 1.0);
    
    // Test ghost exchange indices with local_nz = 0
    if (local_nz == 0) {
        printf("Rank %d: WARNING - Has 0 planes!\n", rank);
        printf("Rank %d: Would access indices:\n", rank);
        printf("  - Send top: local_nz=%d, index for z=local_nz\n", local_nz);
        printf("  - Recv bottom: index for z=0 (ghost)\n");
        printf("  - Send bottom: index for z=1\n");
        printf("  - Recv top: local_nz+1=%d, index for z=local_nz+1\n", local_nz+1);
        printf("  - Buffer has planes 0, 1 (both ghosts when local_nz=0)\n");
    }
    
    MPI_Finalize();
    return 0;
}
