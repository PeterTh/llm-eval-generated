#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * MPI Cahn-Hilliard 3D Solver Snippet
 * 
 * Domain Decomposition: 1D along Z-axis
 * Data Layout: 1D vector with [z][y][x] mapping
 * Ghost Layers: 1 plane on top (z=local_nz+1), 1 plane on bottom (z=0)
 * Real Data: z=1 to z=local_nz
 */

void cahn_hilliard_mpi_snippet(int nx, int ny, int nz, int iterations) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------------------------------------------------------
    // 1. Calculate local Z start and end indices for each rank
    // ---------------------------------------------------------
    
    // Distribute planes as evenly as possible
    int base_local_nz = nz / size;
    int remainder = nz % size;

    // Ranks 0 to remainder-1 get one extra plane
    int local_nz = base_local_nz + (rank < remainder ? 1 : 0);
    
    // Calculate global z_start for this rank (inclusive)
    // Useful for initialization or output mapping
    int global_z_start = 0;
    for (int r = 0; r < rank; ++r) {
        global_z_start += base_local_nz + (r < remainder ? 1 : 0);
    }
    // global_z_end (exclusive) = global_z_start + local_nz;

    // ---------------------------------------------------------
    // 2. Allocating buffers with ghost layers
    // ---------------------------------------------------------

    // We need 1 ghost layer on top and 1 on bottom
    // Local indices: 
    // 0            : Bottom Ghost (received from rank-1)
    // 1 to local_nz: Real Data
    // local_nz + 1 : Top Ghost (received from rank+1)
    
    size_t plane_size = nx * ny;
    size_t local_buffer_size = (local_nz + 2) * plane_size;

    // Vectors for concentration (c), chemical potential (mu), and new concentration (c_new)
    std::vector<double> c_local(local_buffer_size);
    std::vector<double> mu_local(local_buffer_size);
    std::vector<double> c_new_local(local_buffer_size);

    // Helper lambda for local 3D index calculation
    auto idx3_local = [&](int x, int y, int z_local) {
        return z_local * (nx * ny) + y * nx + x;
    };

    // --- Initialization would happen here (mapping global coordinates to local buffer) ---
    // Example: c_local[idx3_local(x, y, z_real_local)] = init_value(x, y, global_z_start + z_real_local - 1);

    // Simulation loop
    for (int t = 0; t < iterations; ++t) {

        // ---------------------------------------------------------
        // 3. Using MPI_Sendrecv to exchange ghost layers
        // ---------------------------------------------------------
        
        int top_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        int bottom_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;

        // Exchange Upward (Tag 0):
        // Send my Top Real Plane (index local_nz) -> Top Neighbor
        // Receive from Bottom Neighbor -> Into my Bottom Ghost (index 0)
        MPI_Sendrecv(
            &c_local[idx3_local(0, 0, local_nz)], plane_size, MPI_DOUBLE, top_neighbor, 0,
            &c_local[idx3_local(0, 0, 0)],        plane_size, MPI_DOUBLE, bottom_neighbor, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // Exchange Downward (Tag 1):
        // Send my Bottom Real Plane (index 1) -> Bottom Neighbor
        // Receive from Top Neighbor -> Into my Top Ghost (index local_nz+1)
        MPI_Sendrecv(
            &c_local[idx3_local(0, 0, 1)],          plane_size, MPI_DOUBLE, bottom_neighbor, 1,
            &c_local[idx3_local(0, 0, local_nz+1)], plane_size, MPI_DOUBLE, top_neighbor, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // ---------------------------------------------------------
        // 4. Handling boundary conditions (Clamped)
        // ---------------------------------------------------------
        
        // At Global Z=0 (Rank 0):
        // The ghost layer at local index 0 is invalid (from MPI_PROC_NULL).
        // Clamped BC: z(-1) = z(0). 
        // Map: local index 0 (ghost) = local index 1 (real).
        if (rank == 0) {
            std::copy(
                c_local.begin() + idx3_local(0, 0, 1),
                c_local.begin() + idx3_local(0, 0, 2),
                c_local.begin() + idx3_local(0, 0, 0)
            );
        }

        // At Global Z=nz-1 (Rank size-1):
        // The ghost layer at local index local_nz+1 is invalid.
        // Clamped BC: z(nz) = z(nz-1).
        // Map: local index local_nz+1 (ghost) = local index local_nz (real).
        if (rank == size - 1) {
            std::copy(
                c_local.begin() + idx3_local(0, 0, local_nz),
                c_local.begin() + idx3_local(0, 0, local_nz + 1),
                c_local.begin() + idx3_local(0, 0, local_nz + 1) // destination
            );
        }

        // --- Compute Chemical Potential (mu) ---
        // Iterate z from 1 to local_nz. 
        // Access z+1, z-1 which are now valid (ghosts or clamped).
        // Store result in mu_local.
        
        // --- Exchange Ghost Layers for mu_local ---
        // (Repeat steps 3 and 4 for mu_local if needed for the next step)
        
        // --- Update Concentration (c_new) ---
        // Iterate z from 1 to local_nz.
        // Compute Laplacian of mu.
        
        // --- Swap buffers ---
        // std::swap(c_local, c_new_local);
    }

    // ---------------------------------------------------------
    // 5. Gathering results back to rank 0
    // ---------------------------------------------------------

    // Prepare global buffer on root
    std::vector<double> c_global;
    if (rank == 0) {
        c_global.resize(nz * plane_size);
    }

    // Prepare arrays for Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    // Calculate recvcounts and displacements for all ranks
    // (Note: In a real app, you might Gather 'local_nz' to avoid recomputing logic)
    for (int r = 0; r < size; ++r) {
        int r_local_nz = base_local_nz + (r < remainder ? 1 : 0);
        recvcounts[r] = r_local_nz * plane_size; 
        displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
    }

    // Gather only the "real" part of local buffers (skipping ghost layers)
    // Send buffer starts at local index 1 (skipping bottom ghost)
    MPI_Gatherv(
        &c_local[idx3_local(0, 0, 1)], // sendbuf
        recvcounts[rank],              // sendcount
        MPI_DOUBLE,                    // sendtype
        c_global.data(),               // recvbuf
        recvcounts.data(),             // recvcounts
        displs.data(),                 // displs
        MPI_DOUBLE,                    // recvtype
        0,                             // root
        MPI_COMM_WORLD                 // comm
    );

    if (rank == 0) {
        // c_global now contains the full 3D field: [z=0..nz-1][y][x]
        std::cout << "Gather complete on rank 0." << std::endl;
    }
}
