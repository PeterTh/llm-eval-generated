#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_offset) {
    for (size_t z = 0; z < local_nz + 2; ++z) {
        long global_z_signed = (long)z_offset + (long)z - 1;
        
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (global_z_signed >= 0) {
                     size_t gz = (size_t)global_z_signed;
                     size_t global_idx = gz * (nx * ny) + y * nx + x;
                     size_t local_idx = idx3(x, y, z, nx, ny);
                     grid[local_idx] = (global_idx % 19) * 1.0;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t nx = 4;
    size_t ny = 4;
    size_t nz = 10;
    
    // Domain Decomposition
    size_t base_nz = nz / size;
    size_t remainder = nz % size;
    size_t local_nz = base_nz + (rank < (int)remainder ? 1 : 0);
    size_t z_offset = rank * base_nz + std::min(rank, (int)remainder);
    
    std::cout << "Rank " << rank << ": local_nz=" << local_nz 
              << ", z_offset=" << z_offset 
              << ", z_range=[" << z_offset << ", " << (z_offset + local_nz - 1) << "]"
              << std::endl;
    
    // Verify decomposition
    std::vector<size_t> all_local_nz(size);
    std::vector<size_t> all_z_offsets(size);
    
    MPI_Gather(&local_nz, 1, MPI_UNSIGNED_LONG, all_local_nz.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&z_offset, 1, MPI_UNSIGNED_LONG, all_z_offsets.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Domain Decomposition Analysis ===" << std::endl;
        size_t total = 0;
        for (int i = 0; i < size; ++i) {
            std::cout << "Rank " << i << ": owns z=[" << all_z_offsets[i] 
                      << ", " << (all_z_offsets[i] + all_local_nz[i] - 1) << "], count=" << all_local_nz[i] << std::endl;
            total += all_local_nz[i];
        }
        std::cout << "Total z-layers: " << total << " (expected: " << nz << ")" << std::endl;
        
        // Check for gaps or overlaps
        bool valid = true;
        for (int i = 0; i < size - 1; ++i) {
            size_t end_i = all_z_offsets[i] + all_local_nz[i] - 1;
            size_t start_next = all_z_offsets[i+1];
            if (end_i + 1 != start_next) {
                std::cout << "ERROR: Gap or overlap between rank " << i << " and " << (i+1) << std::endl;
                valid = false;
            }
        }
        if (valid && total == nz) {
            std::cout << "✓ Decomposition is valid (no gaps/overlaps)" << std::endl;
        }
    }
    
    // Test ghost cell exchange
    size_t local_grid_size = nx * ny * (local_nz + 2);
    std::vector<Real> grid(local_grid_size, -999.0); // Initialize with sentinel
    
    initializeGrid(grid, nx, ny, local_nz, z_offset);
    
    // Print before exchange
    std::cout << "\nRank " << rank << " BEFORE exchange:" << std::endl;
    std::cout << "  Ghost layer 0 (bottom): " << grid[idx3(0, 0, 0, nx, ny)] << std::endl;
    std::cout << "  Owned layer 1 (first): " << grid[idx3(0, 0, 1, nx, ny)] << std::endl;
    std::cout << "  Owned layer " << local_nz << " (last): " << grid[idx3(0, 0, local_nz, nx, ny)] << std::endl;
    std::cout << "  Ghost layer " << (local_nz+1) << " (top): " << grid[idx3(0, 0, local_nz+1, nx, ny)] << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Exchange ghost cells
    MPI_Request reqs[4];
    int nreqs = 0;
    
    if (rank > 0) {
        size_t send_offset = idx3(0, 0, 1, nx, ny);
        MPI_Isend(&grid[send_offset], nx*ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        
        size_t recv_offset = idx3(0, 0, 0, nx, ny);
        MPI_Irecv(&grid[recv_offset], nx*ny, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    
    if (rank < size - 1) {
        size_t send_offset = idx3(0, 0, local_nz, nx, ny);
        MPI_Isend(&grid[send_offset], nx*ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        
        size_t recv_offset = idx3(0, 0, local_nz+1, nx, ny);
        MPI_Irecv(&grid[recv_offset], nx*ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Print after exchange
    std::cout << "\nRank " << rank << " AFTER exchange:" << std::endl;
    std::cout << "  Ghost layer 0 (bottom): " << grid[idx3(0, 0, 0, nx, ny)] << std::endl;
    std::cout << "  Owned layer 1 (first): " << grid[idx3(0, 0, 1, nx, ny)] << std::endl;
    std::cout << "  Owned layer " << local_nz << " (last): " << grid[idx3(0, 0, local_nz, nx, ny)] << std::endl;
    std::cout << "  Ghost layer " << (local_nz+1) << " (top): " << grid[idx3(0, 0, local_nz+1, nx, ny)] << std::endl;
    
    // Verify ghost cells
    MPI_Barrier(MPI_COMM_WORLD);
    bool error = false;
    
    if (rank > 0) {
        // Bottom ghost should match rank-1's last owned layer
        size_t expected_z = z_offset - 1;
        size_t expected_global_idx = expected_z * (nx * ny);
        Real expected_val = (expected_global_idx % 19) * 1.0;
        Real actual_val = grid[idx3(0, 0, 0, nx, ny)];
        if (std::abs(actual_val - expected_val) > 1e-10) {
            std::cout << "ERROR Rank " << rank << ": Bottom ghost mismatch. Expected " << expected_val << " got " << actual_val << std::endl;
            error = true;
        }
    }
    
    if (rank < size - 1) {
        // Top ghost should match rank+1's first owned layer
        size_t expected_z = z_offset + local_nz;
        size_t expected_global_idx = expected_z * (nx * ny);
        Real expected_val = (expected_global_idx % 19) * 1.0;
        Real actual_val = grid[idx3(0, 0, local_nz+1, nx, ny)];
        if (std::abs(actual_val - expected_val) > 1e-10) {
            std::cout << "ERROR Rank " << rank << ": Top ghost mismatch. Expected " << expected_val << " got " << actual_val << std::endl;
            error = true;
        }
    }
    
    int global_error;
    MPI_Reduce(&error, &global_error, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (!global_error) {
            std::cout << "\n✓ Ghost cell exchange validated successfully!" << std::endl;
        } else {
            std::cout << "\n✗ Ghost cell exchange has errors!" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
