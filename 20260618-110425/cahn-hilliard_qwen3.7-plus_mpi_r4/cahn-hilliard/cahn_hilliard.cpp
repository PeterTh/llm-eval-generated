#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Compute local domain size for a given process coordinate
inline size_t localSize(size_t N, int P, int coord) {
    return N / P + (coord < static_cast<int>(N % P) ? 1 : 0);
}

// Compute global offset for a given process coordinate
inline size_t globalOffset(size_t N, int P, int coord) {
    size_t base = N / P;
    size_t extra = N % P;
    return static_cast<size_t>(coord) * base + std::min(static_cast<size_t>(coord), extra);
}

// Halo exchange for all three dimensions
// Array layout: (lnx+2) x (lny+2) x (lnz+2) with interior at [1..lnx]x[1..lny]x[1..lnz]
// dim mapping: dim 0 = z, dim 1 = y, dim 2 = x
void haloExchange(std::vector<double>& data,
                  size_t lnx, size_t lny, size_t lnz,
                  size_t ax, [[maybe_unused]] size_t ay, size_t sxy,
                  const int left[3], const int right[3],
                  MPI_Comm cart_comm,
                  double* buf_send_lo, double* buf_send_hi,
                  double* buf_recv_lo, double* buf_recv_hi) {

    // === Exchange in X dimension (dim 2) ===
    {
        const size_t face_size = lny * lnz;
        // Pack: face at x=1 (lo) and x=lnx (hi)
        for (size_t z = 0; z < lnz; z++) {
            const size_t zrow = (z + 1) * sxy;
            for (size_t y = 0; y < lny; y++) {
                const size_t base = zrow + (y + 1) * ax;
                buf_send_lo[z * lny + y] = data[base + 1];
                buf_send_hi[z * lny + y] = data[base + lnx];
            }
        }

        MPI_Sendrecv(buf_send_hi, face_size, MPI_DOUBLE, right[2], 0,
                     buf_recv_lo, face_size, MPI_DOUBLE, left[2],  0,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf_send_lo, face_size, MPI_DOUBLE, left[2],  1,
                     buf_recv_hi, face_size, MPI_DOUBLE, right[2], 1,
                     cart_comm, MPI_STATUS_IGNORE);

        // Unpack / apply BCs
        if (left[2] != MPI_PROC_NULL) {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t y = 0; y < lny; y++)
                    data[zrow + (y + 1) * ax + 0] = buf_recv_lo[z * lny + y];
            }
        } else {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t y = 0; y < lny; y++)
                    data[zrow + (y + 1) * ax + 0] = data[zrow + (y + 1) * ax + 1];
            }
        }
        if (right[2] != MPI_PROC_NULL) {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t y = 0; y < lny; y++)
                    data[zrow + (y + 1) * ax + lnx + 1] = buf_recv_hi[z * lny + y];
            }
        } else {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t y = 0; y < lny; y++)
                    data[zrow + (y + 1) * ax + lnx + 1] = data[zrow + (y + 1) * ax + lnx];
            }
        }
    }

    // === Exchange in Y dimension (dim 1) ===
    {
        const size_t face_size = lnx * lnz;
        for (size_t z = 0; z < lnz; z++) {
            const size_t zrow = (z + 1) * sxy;
            for (size_t x = 0; x < lnx; x++) {
                buf_send_lo[z * lnx + x] = data[zrow + 1 * ax + (x + 1)];
                buf_send_hi[z * lnx + x] = data[zrow + lny * ax + (x + 1)];
            }
        }

        MPI_Sendrecv(buf_send_hi, face_size, MPI_DOUBLE, right[1], 0,
                     buf_recv_lo, face_size, MPI_DOUBLE, left[1],  0,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf_send_lo, face_size, MPI_DOUBLE, left[1],  1,
                     buf_recv_hi, face_size, MPI_DOUBLE, right[1], 1,
                     cart_comm, MPI_STATUS_IGNORE);

        if (left[1] != MPI_PROC_NULL) {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t x = 0; x < lnx; x++)
                    data[zrow + 0 * ax + (x + 1)] = buf_recv_lo[z * lnx + x];
            }
        } else {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t x = 0; x < lnx; x++)
                    data[zrow + 0 * ax + (x + 1)] = data[zrow + 1 * ax + (x + 1)];
            }
        }
        if (right[1] != MPI_PROC_NULL) {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t x = 0; x < lnx; x++)
                    data[zrow + (lny + 1) * ax + (x + 1)] = buf_recv_hi[z * lnx + x];
            }
        } else {
            for (size_t z = 0; z < lnz; z++) {
                const size_t zrow = (z + 1) * sxy;
                for (size_t x = 0; x < lnx; x++)
                    data[zrow + (lny + 1) * ax + (x + 1)] = data[zrow + lny * ax + (x + 1)];
            }
        }
    }

    // === Exchange in Z dimension (dim 0) ===
    {
        const size_t face_size = lnx * lny;
        for (size_t y = 0; y < lny; y++) {
            const size_t yrow = (y + 1) * ax;
            for (size_t x = 0; x < lnx; x++) {
                buf_send_lo[y * lnx + x] = data[1 * sxy + yrow + (x + 1)];
                buf_send_hi[y * lnx + x] = data[lnz * sxy + yrow + (x + 1)];
            }
        }

        MPI_Sendrecv(buf_send_hi, face_size, MPI_DOUBLE, right[0], 0,
                     buf_recv_lo, face_size, MPI_DOUBLE, left[0],  0,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf_send_lo, face_size, MPI_DOUBLE, left[0],  1,
                     buf_recv_hi, face_size, MPI_DOUBLE, right[0], 1,
                     cart_comm, MPI_STATUS_IGNORE);

        if (left[0] != MPI_PROC_NULL) {
            for (size_t y = 0; y < lny; y++) {
                const size_t yrow = (y + 1) * ax;
                for (size_t x = 0; x < lnx; x++)
                    data[0 * sxy + yrow + (x + 1)] = buf_recv_lo[y * lnx + x];
            }
        } else {
            for (size_t y = 0; y < lny; y++) {
                const size_t yrow = (y + 1) * ax;
                for (size_t x = 0; x < lnx; x++)
                    data[0 * sxy + yrow + (x + 1)] = data[1 * sxy + yrow + (x + 1)];
            }
        }
        if (right[0] != MPI_PROC_NULL) {
            for (size_t y = 0; y < lny; y++) {
                const size_t yrow = (y + 1) * ax;
                for (size_t x = 0; x < lnx; x++)
                    data[(lnz + 1) * sxy + yrow + (x + 1)] = buf_recv_hi[y * lnx + x];
            }
        } else {
            for (size_t y = 0; y < lny; y++) {
                const size_t yrow = (y + 1) * ax;
                for (size_t x = 0; x < lnx; x++)
                    data[(lnz + 1) * sxy + yrow + (x + 1)] = data[lnz * sxy + yrow + (x + 1)];
            }
        }
    }
}

// Compute chemical potential with inlined Laplacian
void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              size_t lnx, size_t lny, size_t lnz,
                              size_t ax, size_t sxy,
                              double dx, double dy, double dz,
                              double gamma, double e_AA, double e_BB, double e_AB) {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    for (size_t z = 1; z <= lnz; z++) {
        const size_t z_sxy  = z * sxy;
        const size_t zp_sxy = (z + 1) * sxy;
        const size_t zn_sxy = (z - 1) * sxy;
        for (size_t y = 1; y <= lny; y++) {
            const size_t y_ax  = y * ax;
            const size_t yp_ax = (y + 1) * ax;
            const size_t yn_ax = (y - 1) * ax;
            for (size_t x = 1; x <= lnx; x++) {
                const size_t idx = z_sxy + y_ax + x;
                const double cv = c[idx];

                const double lap =
                    (c[idx + 1] + c[idx - 1] - 2.0 * cv) * inv_dx2 +
                    (c[z_sxy + yp_ax + x] + c[z_sxy + yn_ax + x] - 2.0 * cv) * inv_dy2 +
                    (c[zp_sxy + y_ax + x] + c[zn_sxy + y_ax + x] - 2.0 * cv) * inv_dz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update with inlined Laplacian
void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        size_t lnx, size_t lny, size_t lnz,
                        size_t ax, size_t sxy,
                        double D, double dt,
                        double dx, double dy, double dz) {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    for (size_t z = 1; z <= lnz; z++) {
        const size_t z_sxy  = z * sxy;
        const size_t zp_sxy = (z + 1) * sxy;
        const size_t zn_sxy = (z - 1) * sxy;
        for (size_t y = 1; y <= lny; y++) {
            const size_t y_ax  = y * ax;
            const size_t yp_ax = (y + 1) * ax;
            const size_t yn_ax = (y - 1) * ax;
            for (size_t x = 1; x <= lnx; x++) {
                const size_t idx = z_sxy + y_ax + x;
                const double mv = mu[idx];

                const double lap =
                    (mu[idx + 1] + mu[idx - 1] - 2.0 * mv) * inv_dx2 +
                    (mu[z_sxy + yp_ax + x] + mu[z_sxy + yn_ax + x] - 2.0 * mv) * inv_dy2 +
                    (mu[zp_sxy + y_ax + x] + mu[zn_sxy + y_ax + x] - 2.0 * mv) * inv_dz2;

                cnew[idx] = cold[idx] + dtD * lap;
            }
        }
    }
}

// Initialize concentration field using global coordinates
void initializeConcentration(double* c, size_t lnx, size_t lny, size_t lnz,
                             size_t ax, size_t sxy,
                             size_t nx, size_t ny, size_t nz,
                             size_t x_off, size_t y_off, size_t z_off) {
    const size_t vol = nx * ny * nz;
    const size_t nxny = nx * ny;

    for (size_t lz = 0; lz < lnz; lz++) {
        const size_t gz = lz + z_off;
        const size_t z_plane = gz * nxny;
        for (size_t ly = 0; ly < lny; ly++) {
            const size_t gy = ly + y_off;
            for (size_t lx = 0; lx < lnx; lx++) {
                const size_t gx = lx + x_off;
                const size_t linear_id = z_plane + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[(lz + 1) * sxy + (ly + 1) * ax + (lx + 1)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Create 3D Cartesian topology
    // dims[0]=z, dims[1]=y, dims[2]=x
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(nprocs, 3, dims);
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int left[3], right[3];
    for (int d = 0; d < 3; d++)
        MPI_Cart_shift(cart_comm, d, 1, &left[d], &right[d]);

    // Local domain sizes and global offsets
    size_t lnx = localSize(nx, dims[2], coords[2]);
    size_t lny = localSize(ny, dims[1], coords[1]);
    size_t lnz = localSize(nz, dims[0], coords[0]);
    size_t x_off = globalOffset(nx, dims[2], coords[2]);
    size_t y_off = globalOffset(ny, dims[1], coords[1]);
    size_t z_off = globalOffset(nz, dims[0], coords[0]);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d (%d x %d x %d)\n", nprocs, dims[0], dims[1], dims[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    // Array dimensions with ghost cells
    const size_t ax = lnx + 2;
    const size_t ay = lny + 2;
    const size_t sxy = ax * ay;
    const size_t total_size = (lnx + 2) * (lny + 2) * (lnz + 2);

    // Allocate arrays
    std::vector<double> cold(total_size, 0.0);
    std::vector<double> cnew(total_size, 0.0);
    std::vector<double> mu(total_size, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), lnx, lny, lnz, ax, sxy, nx, ny, nz, x_off, y_off, z_off);

    // Pre-allocate halo exchange buffers
    const size_t max_face = std::max({lny * lnz, lnx * lnz, lnx * lny});
    std::vector<double> buf_send_lo(max_face);
    std::vector<double> buf_send_hi(max_face);
    std::vector<double> buf_recv_lo(max_face);
    std::vector<double> buf_recv_hi(max_face);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for cold
        haloExchange(cold, lnx, lny, lnz, ax, ay, sxy, left, right, cart_comm,
                     buf_send_lo.data(), buf_send_hi.data(),
                     buf_recv_lo.data(), buf_recv_hi.data());

        // Compute chemical potential
        computeChemicalPotential(cold.data(), mu.data(), lnx, lny, lnz, ax, sxy,
                                 dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Halo exchange for mu
        haloExchange(mu, lnx, lny, lnz, ax, ay, sxy, left, right, cart_comm,
                     buf_send_lo.data(), buf_send_hi.data(),
                     buf_recv_lo.data(), buf_recv_hi.data());

        // Update concentration
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), lnx, lny, lnz, ax, sxy,
                           D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();
    double duration_s = end_time - start_time;
    long duration_ms = static_cast<long>(duration_s * 1000.0);

    // Gather results to rank 0 for output/validation
    const size_t local_interior = lnx * lny * lnz;
    std::vector<double> local_data(local_interior);
    for (size_t z = 0; z < lnz; z++)
        for (size_t y = 0; y < lny; y++)
            for (size_t x = 0; x < lnx; x++)
                local_data[z * lny * lnx + y * lnx + x] =
                    cold[(z + 1) * sxy + (y + 1) * ax + (x + 1)];

    std::vector<double> global_c;
    if (rank == 0) {
        global_c.resize(nx * ny * nz);
        // Unpack rank 0's own data
        for (size_t z = 0; z < lnz; z++)
            for (size_t y = 0; y < lny; y++)
                for (size_t x = 0; x < lnx; x++)
                    global_c[(z + z_off) * nx * ny + (y + y_off) * nx + (x + x_off)] =
                        local_data[z * lny * lnx + y * lnx + x];

        // Receive from other ranks
        for (int r = 1; r < nprocs; r++) {
            int rc[3];
            MPI_Cart_coords(cart_comm, r, 3, rc);
            size_t r_lnx = localSize(nx, dims[2], rc[2]);
            size_t r_lny = localSize(ny, dims[1], rc[1]);
            size_t r_lnz = localSize(nz, dims[0], rc[0]);
            size_t r_xoff = globalOffset(nx, dims[2], rc[2]);
            size_t r_yoff = globalOffset(ny, dims[1], rc[1]);
            size_t r_zoff = globalOffset(nz, dims[0], rc[0]);

            std::vector<double> r_data(r_lnx * r_lny * r_lnz);
            MPI_Recv(r_data.data(), static_cast<int>(r_data.size()), MPI_DOUBLE, r, 42,
                     cart_comm, MPI_STATUS_IGNORE);

            for (size_t z = 0; z < r_lnz; z++)
                for (size_t y = 0; y < r_lny; y++)
                    for (size_t x = 0; x < r_lnx; x++)
                        global_c[(z + r_zoff) * nx * ny + (y + r_yoff) * nx + (x + r_xoff)] =
                            r_data[z * r_lny * r_lnx + y * r_lnx + x];
        }
    } else {
        MPI_Send(local_data.data(), static_cast<int>(local_data.size()), MPI_DOUBLE, 0, 42,
                 cart_comm);
    }

    // Output (rank 0 only)
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        double cellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        double mcups = cellUpdates / duration_s / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(global_c, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_c, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
