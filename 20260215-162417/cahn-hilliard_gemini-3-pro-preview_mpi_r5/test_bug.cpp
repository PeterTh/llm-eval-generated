// Test demonstrating the bug with local_nz = 0
#include <mpi.h>
#include <stdio.h>
#include <vector>

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Bug scenario: nz=2, size=5
    // Ranks 0,1: local_nz=1
    // Ranks 2,3,4: local_nz=0
    int nz = 2;
    int nx = 2, ny = 2;
    int base_local_nz = nz / size;
    int remainder = nz % size;
    int local_nz = base_local_nz + (rank < remainder ? 1 : 0);
    
    size_t plane_size = nx * ny;
    size_t buffer_size = (local_nz + 2) * plane_size;
    std::vector<double> field(buffer_size, -999.0);  // Sentinel value
    
    // Initialize ONLY real data
    for (int z = 1; z <= local_nz; z++) {
        for (size_t i = 0; i < plane_size; i++) {
            field[z * plane_size + i] = rank * 10.0 + z;
        }
    }
    
    printf("Rank %d (local_nz=%d) BEFORE exchange:\n", rank, local_nz);
    printf("  plane[0] (bottom ghost): %.1f\n", field[0]);
    if (local_nz > 0) {
        printf("  plane[1] (first real or top ghost): %.1f\n", field[plane_size]);
        printf("  plane[%d] (top ghost or beyond): %.1f\n", local_nz+1, field[(local_nz+1) * plane_size]);
    } else {
        printf("  plane[1] (top ghost when local_nz=0): %.1f\n", field[plane_size]);
    }
    
    // Ghost exchange
    int top_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int bottom_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    
    printf("Rank %d: Sending to neighbors: top=%d, bottom=%d\n", rank, top_neighbor, bottom_neighbor);
    
    // BUG ANALYSIS:
    // When local_nz=0 (ranks 2,3,4 in this test):
    // - Send upward: &field[idx3(0,0,local_nz)] = &field[idx3(0,0,0)] = &field[0]
    //   This is the BOTTOM GHOST, not real data!
    // - Send downward: &field[idx3(0,0,1)] = &field[plane_size]
    //   This is the TOP GHOST when local_nz=0, not real data!
    //
    // So ranks with local_nz=0 send UNINITIALIZED GHOST DATA
    
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
    
    printf("Rank %d AFTER exchange:\n", rank);
    printf("  plane[0] (bottom ghost): %.1f\n", field[0]);
    if (local_nz > 0) {
        printf("  plane[1]: %.1f\n", field[plane_size]);
        printf("  plane[%d] (top ghost): %.1f\n", local_nz+1, field[(local_nz+1) * plane_size]);
    } else {
        printf("  plane[1] (top ghost when local_nz=0): %.1f\n", field[plane_size]);
    }
    
    // Check for sentinel values (indicates uninitialized data was sent)
    bool has_sentinel = false;
    for (size_t i = 0; i < buffer_size; i++) {
        if (field[i] == -999.0) {
            has_sentinel = true;
            break;
        }
    }
    
    if (!has_sentinel && local_nz == 0) {
        printf("Rank %d: BUG CONFIRMED - Rank with local_nz=0 participated in exchange!\n", rank);
    }
    
    MPI_Finalize();
    return 0;
}
