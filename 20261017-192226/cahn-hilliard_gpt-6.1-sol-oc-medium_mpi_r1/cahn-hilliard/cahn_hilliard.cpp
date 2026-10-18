#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local arrays include one halo cell on every face. Only face neighbors are
// needed, even though two Laplacians are applied in each time step.
struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 0;
    int n[3], start[3], neighbor[6];
    size_t sx, sy, sz;
    MPI_Datatype face[3];

    Domain(MPI_Comm active, const int global[3], const int dims[3]) {
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        int coord[3];
        MPI_Cart_coords(comm, rank, 3, coord);
        for (int d = 0; d < 3; ++d) {
            n[d] = global[d] / dims[d] + (coord[d] < global[d] % dims[d]);
            start[d] = coord[d] * (global[d] / dims[d]) +
                       std::min(coord[d], global[d] % dims[d]);
            MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        }
        sx = static_cast<size_t>(n[0]) + 2;
        sy = static_cast<size_t>(n[1]) + 2;
        sz = static_cast<size_t>(n[2]) + 2;
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        int origin[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d) {
            int subs[3] = {n[2], n[1], n[0]};
            subs[2-d] = 1;
            MPI_Type_create_subarray(3, sizes, subs, origin, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[d]);
            MPI_Type_commit(&face[d]);
        }
    }

    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Comm_free(&comm);
    }

    // Receives and sends have distinct buffers. Physical faces implement the
    // original clamping by copying the adjacent owned face to its halo.
    void beginExchange(std::vector<double>& a, MPI_Request requests[12]) const {
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                int p[3] = {1, 1, 1};
                p[d] = side ? n[d] : 1;
                const size_t owned = idx3(p[0], p[1], p[2], sx, sy);
                p[d] = side ? n[d]+1 : 0;
                const size_t halo = idx3(p[0], p[1], p[2], sx, sy);
                const int f = 2*d+side;
                requests[2*f] = requests[2*f+1] = MPI_REQUEST_NULL;
                if (neighbor[f] == MPI_PROC_NULL) {
                    const size_t stride[3] = {1, sx, sx*sy};
                    for (int z = 0; z < (d == 2 ? 1 : n[2]); ++z)
                        for (int y = 0; y < (d == 1 ? 1 : n[1]); ++y)
                            for (int x = 0; x < (d == 0 ? 1 : n[0]); ++x) {
                                size_t offset = x*stride[0] + y*stride[1] + z*stride[2];
                                a[halo+offset] = a[owned+offset];
                            }
                } else {
                    MPI_Irecv(a.data()+halo, 1, face[d], neighbor[f], 2*d+1-side,
                              comm, &requests[2*f]);
                    MPI_Isend(a.data()+owned, 1, face[d], neighbor[f], f,
                              comm, &requests[2*f+1]);
                }
            }
        }
    }
};

// Select the process grid with the least aggregate face traffic, using as many
// ranks as possible without introducing empty subdomains on thin grids.
int chooseProcessGrid(int ranks, const int global[3], int dims[3]) {
    for (int used = ranks; used > 0; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(used, global[0]); ++x) {
            if (used % x) continue;
            for (int y = 1; y <= std::min(used/x, global[1]); ++y) {
                if ((used/x) % y) continue;
                int z = used/x/y;
                if (z > global[2]) continue;
                double cost = static_cast<double>(x-1)*global[1]*global[2] +
                              static_cast<double>(y-1)*global[0]*global[2] +
                              static_cast<double>(z-1)*global[0]*global[1];
                if (cost < best) {
                    best = cost;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
}

// Compute Laplacian using local face halos.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    (void)nz;
    const size_t xp = x + 1, yp = y + 1, zp = z + 1;
    const size_t xn = x - 1, yn = y - 1, zn = z - 1;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
template<bool interior>
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z < nz-1; ++z) {
        for (size_t y = 1; y < ny-1; ++y) {
            const bool innerRow = z > 1 && z < nz-2 && y > 1 && y < ny-2;
            if (interior && !innerRow) continue;
            for (size_t x = interior ? 2 : 1; x < (interior ? nx-2 : nx-1);
                 x = (!interior && innerRow && x == 1 && nx > 4) ? nx-2 : x+1) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
template<bool interior>
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z < nz-1; ++z) {
        for (size_t y = 1; y < ny-1; ++y) {
            const bool innerRow = z > 1 && z < nz-2 && y > 1 && y < ny-2;
            if (interior && !innerRow) continue;
            for (size_t x = interior ? 2 : 1; x < (interior ? nx-2 : nx-1);
                 x = (!interior && innerRow && x == 1 && nx > 4) ? nx-2 : x+1) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const Domain& domain) {
    const size_t vol = nx * ny * nz;
    
    for (int z = 0; z < domain.n[2]; ++z) {
        for (int y = 0; y < domain.n[1]; ++y) {
            for (int x = 0; x < domain.n[0]; ++x) {
                const size_t idx = idx3(x+1, y+1, z+1, domain.sx, domain.sy);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z+domain.start[2]) * (nx * ny) +
                                         (y+domain.start[1]) * nx + x+domain.start[0];
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const Domain& domain) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    int invalid = 0;
    for (int z = 1; z <= domain.n[2]; ++z)
        for (int y = 1; y <= domain.n[1]; ++y)
            for (int x = 1; x <= domain.n[0]; ++x) {
                double val = c[idx3(x, y, z, domain.sx, domain.sy)];
                invalid |= !std::isfinite(val);
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, domain.comm);
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, domain.comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, domain.comm);
    if (domain.rank == 0) {
        if (invalid) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !invalid && maxVal <= 10.0 && minVal >= -10.0;
}

// Gather only when explicitly requested. Receiving directly into global
// subarrays preserves serial traversal, Kahan sum, samples, and hash ordering.
void printDistributedResults(const std::vector<double>& c, const Domain& domain,
                             const int global[3]) {
    int sizes[3] = {domain.n[2]+2, domain.n[1]+2, domain.n[0]+2};
    int subs[3] = {domain.n[2], domain.n[1], domain.n[0]};
    int origin[3] = {1, 1, 1};
    MPI_Datatype owned;
    MPI_Type_create_subarray(3, sizes, subs, origin, MPI_ORDER_C, MPI_DOUBLE, &owned);
    MPI_Type_commit(&owned);
    int metadata[6] = {domain.n[0], domain.n[1], domain.n[2],
                       domain.start[0], domain.start[1], domain.start[2]};
    if (domain.rank != 0) {
        MPI_Send(metadata, 6, MPI_INT, 0, 10, domain.comm);
        MPI_Send(c.data(), 1, owned, 0, 11, domain.comm);
    } else {
        std::vector<double> result(static_cast<size_t>(global[0])*global[1]*global[2]);
        for (int p = 0; p < domain.size; ++p) {
            if (p) MPI_Recv(metadata, 6, MPI_INT, p, 10, domain.comm, MPI_STATUS_IGNORE);
            int gs[3] = {global[2], global[1], global[0]};
            int ns[3] = {metadata[2], metadata[1], metadata[0]};
            int starts[3] = {metadata[5], metadata[4], metadata[3]};
            MPI_Datatype block;
            MPI_Type_create_subarray(3, gs, ns, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            if (p) MPI_Recv(result.data(), 1, block, p, 11, domain.comm, MPI_STATUS_IGNORE);
            else {
                for (int z = 0; z < domain.n[2]; ++z)
                    for (int y = 0; y < domain.n[1]; ++y)
                        std::copy_n(c.data()+idx3(1,y+1,z+1,domain.sx,domain.sy), domain.n[0],
                                    result.data()+idx3(domain.start[0], y+domain.start[1],
                                                      z+domain.start[2], global[0], global[1]));
            }
            MPI_Type_free(&block);
        }
        print_results(result, "Concentration");
    }
    MPI_Type_free(&owned);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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

    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > std::numeric_limits<size_t>::max()/ny ||
        nx*ny > std::numeric_limits<size_t>::max()/nz) {
        if (rank == 0) fprintf(stderr, "Invalid or unsupported grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const int global[3] = {static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz)};
    int dims[3];
    const int used = chooseProcessGrid(ranks, global, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    if (rank < used) {
        Domain domain(active, global, dims);
        MPI_Comm_free(&active);

        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
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

        const size_t gridSize = nx * ny * nz;

        // Allocate only the local subdomain and its halos.
        const size_t localSize = domain.sx * domain.sy * domain.sz;
        std::vector<double> cold(localSize);
        std::vector<double> cnew(localSize);
        std::vector<double> mu(localSize);

        if (rank == 0) printf("Initializing concentration field...\n");
        initializeConcentration(cold, nx, ny, nz, domain);

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(domain.comm);
        double start = MPI_Wtime();

        for (int t = 0; t < iterations; ++t) {
            MPI_Request requests[12];
            domain.beginExchange(cold, requests);
            computeChemicalPotential<true>(cold, mu, domain.sx, domain.sy, domain.sz, dx, dy, dz,
                                           gamma, e_AA, e_BB, e_AB);
            MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
            computeChemicalPotential<false>(cold, mu, domain.sx, domain.sy, domain.sz, dx, dy, dz,
                                            gamma, e_AA, e_BB, e_AB);

            domain.beginExchange(mu, requests);
            cahnHilliardUpdate<true>(cnew, cold, mu, domain.sx, domain.sy, domain.sz, D, dt, dx, dy, dz);
            MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
            cahnHilliardUpdate<false>(cnew, cold, mu, domain.sx, domain.sy, domain.sz, D, dt, dx, dy, dz);

            std::swap(cold, cnew);
        }

        double duration = MPI_Wtime() - start;
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);

        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
            double cellUpdates = static_cast<double>(gridSize) * iterations;
            double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            printDistributedResults(cold, domain, global);
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateResult(cold, domain);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    } // Free MPI datatypes and communicator before finalizing.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
