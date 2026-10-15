#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local slab: planes [z0, z0+nl) of the global grid stored at local planes 1..nl,
// with halo planes 0 and nl+1.
void initializeSlab(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t z0, const size_t nl) {
    const size_t plane = nx * ny;
    for (size_t lz = 0; lz < nl; ++lz) {
        Real* p = &grid[(lz + 1) * plane];
        const size_t base = (z0 + lz) * plane;
        for (size_t i = 0; i < plane; ++i) {
            p[i] = ((base + i) % 19) * 1.0;
        }
    }
}

// Update one local plane lz (1-based local index) whose global z is interior
inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                         const size_t nx, const size_t ny, const size_t lz) {
    const size_t plane = nx * ny;
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = lz * plane + y * nx;
        const Real* c = in + row;
        const Real* fr = in + row - nx;
        const Real* bk = in + row + nx;
        const Real* bo = in + row - plane;
        const Real* tp = in + row + plane;
        Real* o = out + row;
        for (size_t x = 1; x < nx - 1; ++x) {
            o[x] = (c[x] + c[x-1] + c[x+1] + fr[x] + bk[x] + bo[x] + tp[x]) / 7.0;
        }
    }
}

// Distributed validation over each rank's owned cells; result identical on all ranks
bool validateResult(const Real* data, const size_t n, const bool active, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    if (active) {
        for (size_t i = 0; i < n; ++i) {
            const Real val = data[i];
            if (std::isnan(val) || std::isinf(val)) bad = 1;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    int gbad;
    double gmin, gmax;
    MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (gbad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 1e6 || gmin < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (rank == 0) printUsage(argv[0]);
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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D decomposition along Z; only ranks that own at least one plane are active
    const size_t plane = nx * ny;
    const int nactive = (int)std::min<size_t>((size_t)nprocs, std::max<size_t>(nz, 1));
    const bool active = rank < nactive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &comm);

    auto zStart = [&](int r) { return (nz / nactive) * r + std::min<size_t>(r, nz % nactive); };

    size_t z0 = 0, nl = 0;
    std::vector<Real> grid1, grid2;
    if (rank == 0) printf("Initializing grid...\n");
    if (active) {
        z0 = zStart(rank);
        nl = zStart(rank + 1) - z0;
        grid1.resize((nl + 2) * plane);
        initializeSlab(grid1, nx, ny, z0, nl);
        // Boundary cells never change, so both buffers start identical
        grid2 = grid1;
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        const int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        const int down = (rank < nactive - 1) ? rank + 1 : MPI_PROC_NULL;
        // Local plane lz is updated iff its global z is interior
        auto interior = [&](size_t lz) { size_t gz = z0 + lz - 1; return gz >= 1 && gz + 1 < nz; };
        std::vector<Real>* cur = &grid1;
        std::vector<Real>* nxt = &grid2;
        for (int iter = 0; iter < iterations; ++iter) {
            Real* in = cur->data();
            Real* out = nxt->data();
            MPI_Request req[4];
            // Halo exchange of boundary planes of the input
            MPI_Irecv(in, (int)plane, MPI_DOUBLE, up, 0, comm, &req[0]);
            MPI_Irecv(in + (nl + 1) * plane, (int)plane, MPI_DOUBLE, down, 1, comm, &req[1]);
            MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, up, 1, comm, &req[2]);
            MPI_Isend(in + nl * plane, (int)plane, MPI_DOUBLE, down, 0, comm, &req[3]);

            // Overlap: planes not touching halos
            for (size_t lz = 2; lz + 1 <= nl; ++lz) {
                if (interior(lz)) stencilPlane(in, out, nx, ny, lz);
            }
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            if (interior(1)) stencilPlane(in, out, nx, ny, 1);
            if (nl > 1 && interior(nl)) stencilPlane(in, out, nx, ny, nl);
            std::swap(cur, nxt);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long ms = duration.count();
    long long maxms;
    MPI_Allreduce(&ms, &maxms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)maxms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* owned = active ? finalGrid.data() + plane : nullptr;

    // Print results for external validation: gather full grid on rank 0
    if (printResults) {
        const size_t maxChunk = (size_t)1 << 27;
        if (rank == 0) {
            std::vector<Real> full(nx * ny * nz);
            memcpy(full.data(), owned, nl * plane * sizeof(Real));
            for (int r = 1; r < nactive; ++r) {
                size_t off = zStart(r) * plane;
                size_t cnt = (zStart(r + 1) - zStart(r)) * plane;
                for (size_t done = 0; done < cnt; done += maxChunk) {
                    int c = (int)std::min(maxChunk, cnt - done);
                    MPI_Recv(full.data() + off + done, c, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            print_results(full, "Grid");
        } else if (active) {
            size_t cnt = nl * plane;
            for (size_t done = 0; done < cnt; done += maxChunk) {
                int c = (int)std::min(maxChunk, cnt - done);
                MPI_Send(owned + done, c, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(owned, nl * plane, active, MPI_COMM_WORLD, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    if (active) MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
