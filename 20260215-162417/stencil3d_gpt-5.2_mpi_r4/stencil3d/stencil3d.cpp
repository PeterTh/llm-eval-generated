#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Global 3D index calculation (row-major: x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nxh, const size_t nyh) noexcept {
    return z * (nxh * nyh) + y * nxh + x;
}

static inline void decompose_1d(const size_t n, const int p, const int coord, size_t& local_n, size_t& start) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    local_n = base + (static_cast<size_t>(coord) < rem ? 1u : 0u);
    start = base * static_cast<size_t>(coord) + std::min(static_cast<size_t>(coord), rem);
}

static inline void initializeGridLocal(std::vector<Real>& grid,
                                      const size_t nxh, const size_t nyh, const size_t nzh,
                                      const size_t nx, const size_t ny,
                                      const size_t x0, const size_t y0, const size_t z0,
                                      const size_t lx, const size_t ly, const size_t lz) {
    (void)nzh;
    for (size_t z = 1; z <= lz; ++z) {
        const size_t gz = z0 + (z - 1);
        for (size_t y = 1; y <= ly; ++y) {
            const size_t gy = y0 + (y - 1);
            const size_t row_base = idx3_local(1, y, z, nxh, nyh);
            for (size_t x = 1; x <= lx; ++x) {
                const size_t gx = x0 + (x - 1);
                const size_t gidx = idx3(gx, gy, gz, nx, ny);
                grid[row_base + (x - 1)] = static_cast<Real>(gidx % 19) * 1.0;
            }
        }
    }
}

static inline bool validateResultDistributed(const std::vector<Real>& local,
                                            const size_t nxh, const size_t nyh,
                                            const size_t lx, const size_t ly, const size_t lz,
                                            MPI_Comm comm, const int rank) {
    bool local_ok = true;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();

    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            const size_t base = idx3_local(1, y, z, nxh, nyh);
            for (size_t x = 0; x < lx; ++x) {
                const Real v = local[base + x];
                if (std::isnan(v) || std::isinf(v)) {
                    local_ok = false;
                }
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }

    int ok_in = local_ok ? 1 : 0;
    int ok_out = 0;
    MPI_Allreduce(&ok_in, &ok_out, 1, MPI_INT, MPI_LAND, comm);

    Real gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
        if (gmax > 1e6 || gmin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
        if (!ok_out) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    return ok_out != 0;
}

static inline void copy_global_boundary_planes(const std::vector<Real>& in, std::vector<Real>& out,
                                              const size_t nxh, const size_t nyh,
                                              const size_t nx, const size_t ny, const size_t nz,
                                              const size_t x0, const size_t y0, const size_t z0,
                                              const size_t lx, const size_t ly, const size_t lz) {
    (void)nz;

    // x == 0 plane
    if (x0 == 0 && lx > 0) {
        const size_t i = 1;
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                const size_t id = idx3_local(i, y, z, nxh, nyh);
                out[id] = in[id];
            }
        }
    }
    // x == nx-1 plane
    if (x0 + lx == nx && lx > 0) {
        const size_t i = lx;
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                const size_t id = idx3_local(i, y, z, nxh, nyh);
                out[id] = in[id];
            }
        }
    }

    // y == 0 plane
    if (y0 == 0 && ly > 0) {
        const size_t j = 1;
        for (size_t z = 1; z <= lz; ++z) {
            const size_t base = idx3_local(1, j, z, nxh, nyh);
            const size_t base_in = base;
            for (size_t x = 0; x < lx; ++x) {
                out[base + x] = in[base_in + x];
            }
        }
    }
    // y == ny-1 plane
    if (y0 + ly == ny && ly > 0) {
        const size_t j = ly;
        for (size_t z = 1; z <= lz; ++z) {
            const size_t base = idx3_local(1, j, z, nxh, nyh);
            const size_t base_in = base;
            for (size_t x = 0; x < lx; ++x) {
                out[base + x] = in[base_in + x];
            }
        }
    }

    // z == 0 plane
    if (z0 == 0 && lz > 0) {
        const size_t k = 1;
        for (size_t y = 1; y <= ly; ++y) {
            const size_t base = idx3_local(1, y, k, nxh, nyh);
            for (size_t x = 0; x < lx; ++x) {
                out[base + x] = in[base + x];
            }
        }
    }
    // z == nz-1 plane
    if (z0 + lz == nz && lz > 0) {
        const size_t k = lz;
        for (size_t y = 1; y <= ly; ++y) {
            const size_t base = idx3_local(1, y, k, nxh, nyh);
            for (size_t x = 0; x < lx; ++x) {
                out[base + x] = in[base + x];
            }
        }
    }
}

static inline void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation (gathers to rank 0)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    int parse_ok = 1;
    int want_help = 0;

    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                want_help = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parse_ok = 0;
            }
        }

        if (want_help) {
            printUsage(argv[0]);
        }
        if (!parse_ok) {
            printUsage(argv[0]);
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;

        if (nx < 1 || ny < 1 || nz < 1 || iterations < 0) {
            printf("Invalid parameters\n");
            parse_ok = 0;
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&want_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok || want_help) {
        MPI_Finalize();
        return parse_ok ? 0 : 1;
    }

    // Broadcast configuration
    uint64_t nx_u = static_cast<uint64_t>(nx);
    uint64_t ny_u = static_cast<uint64_t>(ny);
    uint64_t nz_u = static_cast<uint64_t>(nz);
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;

    MPI_Bcast(&nx_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    nx = static_cast<size_t>(nx_u);
    ny = static_cast<size_t>(ny_u);
    nz = static_cast<size_t>(nz_u);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    // Create 3D Cartesian topology
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims);
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int cart_rank = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

    size_t lx = 0, ly = 0, lz = 0;
    size_t x0 = 0, y0 = 0, z0 = 0;
    decompose_1d(nx, dims[2], coords[2], lx, x0);
    decompose_1d(ny, dims[1], coords[1], ly, y0);
    decompose_1d(nz, dims[0], coords[0], lz, z0);

    if (lx == 0 || ly == 0 || lz == 0) {
        if (world_rank == 0) {
            printf("Error: MPI decomposition produced an empty subdomain. Reduce MPI ranks or increase grid size.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const size_t nxh = lx + 2;
    const size_t nyh = ly + 2;
    const size_t nzh = lz + 2;

    if (world_rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Cart dims (Z,Y,X): %d x %d x %d\n", dims[0], dims[1], dims[2]);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local grids with halos (double buffering)
    std::vector<Real> grid1(nxh * nyh * nzh, 0.0);
    std::vector<Real> grid2(nxh * nyh * nzh, 0.0);

    // Initialize local interior
    if (world_rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGridLocal(grid1, nxh, nyh, nzh, nx, ny, x0, y0, z0, lx, ly, lz);

    // Neighbor ranks
    int nbr_zm = MPI_PROC_NULL, nbr_zp = MPI_PROC_NULL;
    int nbr_ym = MPI_PROC_NULL, nbr_yp = MPI_PROC_NULL;
    int nbr_xm = MPI_PROC_NULL, nbr_xp = MPI_PROC_NULL;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_zm, &nbr_zp);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_ym, &nbr_yp);
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_xm, &nbr_xp);

    // Halo exchange datatypes (subarrays inside the local grid including halos)
    MPI_Datatype send_xm, recv_xm, send_xp, recv_xp;
    MPI_Datatype send_ym, recv_ym, send_yp, recv_yp;
    MPI_Datatype send_zm, recv_zm, send_zp, recv_zp;

    const int sizes[3] = {static_cast<int>(nzh), static_cast<int>(nyh), static_cast<int>(nxh)};

    auto make_sub = [&](const int subs[3], const int starts[3], MPI_Datatype& t) {
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
        MPI_Type_commit(&t);
    };

    {
        const int subs_x[3] = {static_cast<int>(lz), static_cast<int>(ly), 1};
        const int subs_y[3] = {static_cast<int>(lz), 1, static_cast<int>(lx)};
        const int subs_z[3] = {1, static_cast<int>(ly), static_cast<int>(lx)};

        const int sxm_s[3] = {1, 1, 1};
        const int sxm_r[3] = {1, 1, 0};
        const int sxp_s[3] = {1, 1, static_cast<int>(lx)};
        const int sxp_r[3] = {1, 1, static_cast<int>(lx + 1)};

        const int sym_s[3] = {1, 1, 1};
        const int sym_r[3] = {1, 0, 1};
        const int syp_s[3] = {1, static_cast<int>(ly), 1};
        const int syp_r[3] = {1, static_cast<int>(ly + 1), 1};

        const int szm_s[3] = {1, 1, 1};
        const int szm_r[3] = {0, 1, 1};
        const int szp_s[3] = {static_cast<int>(lz), 1, 1};
        const int szp_r[3] = {static_cast<int>(lz + 1), 1, 1};

        make_sub(subs_x, sxm_s, send_xm);
        make_sub(subs_x, sxm_r, recv_xm);
        make_sub(subs_x, sxp_s, send_xp);
        make_sub(subs_x, sxp_r, recv_xp);

        make_sub(subs_y, sym_s, send_ym);
        make_sub(subs_y, sym_r, recv_ym);
        make_sub(subs_y, syp_s, send_yp);
        make_sub(subs_y, syp_r, recv_yp);

        make_sub(subs_z, szm_s, send_zm);
        make_sub(subs_z, szm_r, recv_zm);
        make_sub(subs_z, szp_s, send_zp);
        make_sub(subs_z, szp_r, recv_zp);
    }

    constexpr int TAG_X_MINUS = 100;
    constexpr int TAG_X_PLUS = 101;
    constexpr int TAG_Y_MINUS = 110;
    constexpr int TAG_Y_PLUS = 111;
    constexpr int TAG_Z_MINUS = 120;
    constexpr int TAG_Z_PLUS = 121;

    auto halo_exchange = [&](std::vector<Real>& in) {
        MPI_Request reqs[12];
        int r = 0;

        // Receives first
        if (nbr_xm != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_xm, nbr_xm, TAG_X_PLUS, cart_comm, &reqs[r++]);
        if (nbr_xp != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_xp, nbr_xp, TAG_X_MINUS, cart_comm, &reqs[r++]);
        if (nbr_ym != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_ym, nbr_ym, TAG_Y_PLUS, cart_comm, &reqs[r++]);
        if (nbr_yp != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_yp, nbr_yp, TAG_Y_MINUS, cart_comm, &reqs[r++]);
        if (nbr_zm != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_zm, nbr_zm, TAG_Z_PLUS, cart_comm, &reqs[r++]);
        if (nbr_zp != MPI_PROC_NULL) MPI_Irecv(in.data(), 1, recv_zp, nbr_zp, TAG_Z_MINUS, cart_comm, &reqs[r++]);

        // Sends
        if (nbr_xm != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_xm, nbr_xm, TAG_X_MINUS, cart_comm, &reqs[r++]);
        if (nbr_xp != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_xp, nbr_xp, TAG_X_PLUS, cart_comm, &reqs[r++]);
        if (nbr_ym != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_ym, nbr_ym, TAG_Y_MINUS, cart_comm, &reqs[r++]);
        if (nbr_yp != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_yp, nbr_yp, TAG_Y_PLUS, cart_comm, &reqs[r++]);
        if (nbr_zm != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_zm, nbr_zm, TAG_Z_MINUS, cart_comm, &reqs[r++]);
        if (nbr_zp != MPI_PROC_NULL) MPI_Isend(in.data(), 1, send_zp, nbr_zp, TAG_Z_PLUS, cart_comm, &reqs[r++]);

        if (r > 0) {
            MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
        }
    };

    // Precompute local index ranges corresponding to global interior (exclude global boundaries)
    auto compute_local_range = [](size_t start0, size_t local_n, size_t n_global, size_t& l_begin, size_t& l_end, bool& nonempty) {
        if (n_global <= 2 || local_n == 0) {
            nonempty = false;
            return;
        }
        const size_t g_begin = std::max<size_t>(1, start0);
        const size_t g_end = std::min<size_t>(n_global - 2, start0 + local_n - 1);
        if (g_begin > g_end) {
            nonempty = false;
            return;
        }
        l_begin = (g_begin - start0) + 1;
        l_end = (g_end - start0) + 1;
        nonempty = true;
    };

    size_t ix_b = 0, ix_e = 0, iy_b = 0, iy_e = 0, iz_b = 0, iz_e = 0;
    bool x_has = false, y_has = false, z_has = false;
    compute_local_range(x0, lx, nx, ix_b, ix_e, x_has);
    compute_local_range(y0, ly, ny, iy_b, iy_e, y_has);
    compute_local_range(z0, lz, nz, iz_b, iz_e, z_has);

    if (world_rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(cart_comm);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;

        halo_exchange(in);

        // Compute global interior points
        if (x_has && y_has && z_has) {
            const size_t sx = 1;
            const size_t sy = nxh;
            const size_t sz = nxh * nyh;

            const Real* __restrict in_ptr = in.data();
            Real* __restrict out_ptr = out.data();

            for (size_t z = iz_b; z <= iz_e; ++z) {
                const size_t zoff = z * sz;
                for (size_t y = iy_b; y <= iy_e; ++y) {
                    size_t idx = zoff + y * sy + ix_b;
                    for (size_t x = ix_b; x <= ix_e; ++x, ++idx) {
                        const Real c = in_ptr[idx];
                        out_ptr[idx] = (c + in_ptr[idx - sx] + in_ptr[idx + sx] + in_ptr[idx - sy] + in_ptr[idx + sy] + in_ptr[idx - sz] + in_ptr[idx + sz]) / 7.0;
                    }
                }
            }
        }

        // Copy global boundary planes unchanged
        copy_global_boundary_planes(in, out, nxh, nyh, nx, ny, nz, x0, y0, z0, lx, ly, lz);
    }

    MPI_Barrier(cart_comm);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double local_sec = std::chrono::duration<double>(t1 - t0).count();
    double max_sec = 0.0;
    MPI_Allreduce(&local_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_sec * 1000.0));
        const double cellUpdates = static_cast<double>((nx > 2 ? (nx - 2) : 0) * (ny > 2 ? (ny - 2) : 0) * (nz > 2 ? (nz - 2) : 0)) * static_cast<double>(iterations);
        const double mcups = (max_sec > 0.0) ? (cellUpdates / max_sec / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Final grid
    std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults) {
        if (world_rank == 0) {
            printf("Gathering results to rank 0...\n");
        }

        std::vector<Real> global;
        if (world_rank == 0) {
            global.resize(nx * ny * nz);
        }

        // Pack local interior contiguous
        std::vector<Real> sendbuf(lx * ly * lz);
        size_t p = 0;
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                const size_t base = idx3_local(1, y, z, nxh, nyh);
                std::memcpy(sendbuf.data() + p, finalLocal.data() + base, lx * sizeof(Real));
                p += lx;
            }
        }

        constexpr int GATHER_TAG = 777;

        if (world_rank == 0) {
            // Copy rank 0 into global
            for (size_t z = 0; z < lz; ++z) {
                for (size_t y = 0; y < ly; ++y) {
                    const size_t goff = idx3(x0, y0 + y, z0 + z, nx, ny);
                    const size_t soff = (z * ly + y) * lx;
                    std::memcpy(global.data() + goff, sendbuf.data() + soff, lx * sizeof(Real));
                }
            }

            std::vector<MPI_Request> reqs;
            reqs.reserve(static_cast<size_t>(world_size > 1 ? world_size - 1 : 0));

            for (int r = 1; r < world_size; ++r) {
                int c[3];
                MPI_Cart_coords(cart_comm, r, 3, c);

                size_t r_lx = 0, r_ly = 0, r_lz = 0;
                size_t r_x0 = 0, r_y0 = 0, r_z0 = 0;
                decompose_1d(nx, dims[2], c[2], r_lx, r_x0);
                decompose_1d(ny, dims[1], c[1], r_ly, r_y0);
                decompose_1d(nz, dims[0], c[0], r_lz, r_z0);

                const int gsizes[3] = {static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx)};
                const int subs[3] = {static_cast<int>(r_lz), static_cast<int>(r_ly), static_cast<int>(r_lx)};
                const int starts[3] = {static_cast<int>(r_z0), static_cast<int>(r_y0), static_cast<int>(r_x0)};
                MPI_Datatype rtype;
                MPI_Type_create_subarray(3, gsizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &rtype);
                MPI_Type_commit(&rtype);

                MPI_Request rq;
                MPI_Irecv(global.data(), 1, rtype, r, GATHER_TAG, cart_comm, &rq);
                reqs.push_back(rq);

                MPI_Type_free(&rtype);
            }

            if (!reqs.empty()) {
                MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
            }

            print_results(global, "Grid");
        } else {
            MPI_Send(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE, 0, GATHER_TAG, cart_comm);
        }
    }

    if (validate) {
        if (world_rank == 0) {
            printf("Validating result...\n");
        }
        const bool ok = validateResultDistributed(finalLocal, nxh, nyh, lx, ly, lz, cart_comm, world_rank);
        if (world_rank == 0) {
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }

        MPI_Type_free(&send_xm);
        MPI_Type_free(&recv_xm);
        MPI_Type_free(&send_xp);
        MPI_Type_free(&recv_xp);
        MPI_Type_free(&send_ym);
        MPI_Type_free(&recv_ym);
        MPI_Type_free(&send_yp);
        MPI_Type_free(&recv_yp);
        MPI_Type_free(&send_zm);
        MPI_Type_free(&recv_zm);
        MPI_Type_free(&send_zp);
        MPI_Type_free(&recv_zp);
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();

        return ok ? 0 : 1;
    }

    MPI_Type_free(&send_xm);
    MPI_Type_free(&recv_xm);
    MPI_Type_free(&send_xp);
    MPI_Type_free(&recv_xp);
    MPI_Type_free(&send_ym);
    MPI_Type_free(&recv_ym);
    MPI_Type_free(&send_yp);
    MPI_Type_free(&recv_yp);
    MPI_Type_free(&send_zm);
    MPI_Type_free(&recv_zm);
    MPI_Type_free(&send_zp);
    MPI_Type_free(&recv_zp);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
