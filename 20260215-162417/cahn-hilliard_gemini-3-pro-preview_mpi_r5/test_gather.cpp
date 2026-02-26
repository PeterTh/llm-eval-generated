#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Test various nz/size ratios
    int test_cases[][2] = {
        {8, 4},   // Normal: 2,2,2,2
        {7, 4},   // Uneven: 2,2,2,1
        {3, 5},   // Empty ranks: 1,1,1,0,0
        {2, 4},   // More empty: 1,1,0,0
        {1, 3}    // Mostly empty: 1,0,0
    };
    
    for (int tc = 0; tc < 5; tc++) {
        int nz = test_cases[tc][0];
        int test_size = test_cases[tc][1];
        
        if (size != test_size) continue;
        
        int nx = 2, ny = 2;
        int base_local_nz = nz / size;
        int remainder = nz % size;
        int local_nz = base_local_nz + (rank < remainder ? 1 : 0);
        
        size_t plane_size = nx * ny;
        size_t buffer_size = (local_nz + 2) * plane_size;
        std::vector<double> field(buffer_size, 0.0);
        
        // Initialize with unique values
        int z_start = 0;
        for (int r = 0; r < rank; r++) {
            z_start += base_local_nz + (r < remainder ? 1 : 0);
        }
        
        for (int z = 1; z <= local_nz; z++) {
            int global_z = z_start + (z - 1);
            for (size_t i = 0; i < plane_size; i++) {
                field[idx3(i % nx, (i / nx) % ny, z, nx, ny)] = global_z * 100.0 + i;
            }
        }
        
        // Gather data
        std::vector<double> global_data;
        if (rank == 0) {
            global_data.resize(nz * plane_size);
        }
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = local_nz * plane_size;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < size; r++) {
                displs[r] = displs[r-1] + recvcounts[r-1];
            }
        }
        
        MPI_Gatherv(&field[plane_size], my_count, MPI_DOUBLE, 
                   global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
        
        // Verify
        if (rank == 0) {
            bool correct = true;
            for (int gz = 0; gz < nz; gz++) {
                for (size_t i = 0; i < plane_size; i++) {
                    double expected = gz * 100.0 + i;
                    double actual = global_data[gz * plane_size + i];
                    if (actual != expected) {
                        printf("ERROR: nz=%d, size=%d, global_z=%d, i=%zu: expected %.1f, got %.1f\n",
                               nz, size, gz, i, expected, actual);
                        correct = false;
                    }
                }
            }
            if (correct) {
                printf("PASS: nz=%d, size=%d - data gathered correctly\n", nz, size);
            } else {
                printf("FAIL: nz=%d, size=%d\n", nz, size);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
