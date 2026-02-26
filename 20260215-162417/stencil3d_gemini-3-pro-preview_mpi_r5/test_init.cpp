#include <iostream>
#include <vector>

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_offset, const size_t global_nz) {
    for (size_t z = 0; z < local_nz + 2; ++z) {
        long global_z_signed = (long)z_offset + (long)z - 1;
        
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (global_z_signed >= 0) {
                     size_t gz = (size_t)global_z_signed;
                     size_t global_idx = gz * (nx * ny) + y * nx + x;
                     size_t local_idx = idx3(x, y, z, nx, ny);
                     grid[local_idx] = (global_idx % 19) * 1.0;
                     
                     if (x == 0 && y == 0) {
                         std::cout << "  z=" << z << " (global_z=" << gz << "): value=" << grid[local_idx] << std::endl;
                     }
                }
            }
        }
    }
}

int main() {
    size_t nx = 4, ny = 4, nz = 10;
    int size = 3;
    
    for (int rank = 0; rank < size; ++rank) {
        size_t base_nz = nz / size;
        size_t remainder = nz % size;
        size_t local_nz = base_nz + (rank < (int)remainder ? 1 : 0);
        size_t z_offset = rank * base_nz + std::min(rank, (int)remainder);
        
        std::cout << "\nRank " << rank << " (local_nz=" << local_nz << ", z_offset=" << z_offset << "):" << std::endl;
        
        size_t local_grid_size = nx * ny * (local_nz + 2);
        std::vector<Real> grid(local_grid_size, -999.0);
        
        initializeGrid(grid, nx, ny, local_nz, z_offset, nz);
        
        // Check if any values are being initialized outside valid global range
        for (size_t z = 0; z < local_nz + 2; ++z) {
            long global_z_signed = (long)z_offset + (long)z - 1;
            if (global_z_signed >= (long)nz && grid[idx3(0, 0, z, nx, ny)] != -999.0) {
                std::cout << "ERROR: Layer " << z << " (global_z=" << global_z_signed << ") initialized beyond nz=" << nz << std::endl;
            }
        }
    }
    
    return 0;
}
