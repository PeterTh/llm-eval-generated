#include <iostream>
#include <vector>

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

int main() {
    size_t nx = 4, ny = 4;
    
    // Test different nz values
    std::vector<size_t> test_nz = {10, 11, 12, 15, 16, 17, 20};
    std::vector<int> test_sizes = {3, 4, 5};
    
    for (size_t nz : test_nz) {
        for (int size : test_sizes) {
            std::cout << "\n=== Testing nz=" << nz << ", size=" << size << " ===" << std::endl;
            
            size_t total_assigned = 0;
            size_t last_global_z = 0;
            
            for (int rank = 0; rank < size; ++rank) {
                size_t base_nz = nz / size;
                size_t remainder = nz % size;
                size_t local_nz = base_nz + (rank < (int)remainder ? 1 : 0);
                size_t z_offset = rank * base_nz + std::min(rank, (int)remainder);
                
                total_assigned += local_nz;
                size_t end_z = z_offset + local_nz - 1;
                
                std::cout << "  Rank " << rank << ": z=[" << z_offset << ".." << end_z << "], count=" << local_nz;
                
                // Check for gap
                if (rank > 0 && z_offset != last_global_z + 1) {
                    std::cout << " *** GAP between rank " << (rank-1) << " and " << rank << " ***";
                }
                
                // Check upper ghost initialization
                long top_ghost_z = (long)z_offset + (long)local_nz;
                if (rank == size - 1 && top_ghost_z >= (long)nz) {
                    std::cout << " *** UPPER GHOST OUT OF BOUNDS (z=" << top_ghost_z << " >= nz=" << nz << ") ***";
                }
                
                std::cout << std::endl;
                last_global_z = end_z;
            }
            
            if (total_assigned != nz) {
                std::cout << "  ERROR: Total assigned " << total_assigned << " != nz " << nz << std::endl;
            }
        }
    }
    
    return 0;
}
