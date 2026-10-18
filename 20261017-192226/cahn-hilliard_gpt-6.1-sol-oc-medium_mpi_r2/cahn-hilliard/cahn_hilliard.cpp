#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// The six face halos supply both neighboring ranks and clamped physical edges.
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

// A box is half-open in each coordinate, excluding halos.
struct Box {
    size_t x0, x1, y0, y1, z0, z1;
};

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const Box box) {
    for (size_t z = box.z0; z < box.z1; ++z) {
        for (size_t y = box.y0; y < box.y1; ++y) {
            for (size_t x = box.x0; x < box.x1; ++x) {
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
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                         const double D, const double dt, const double dx, const double dy, const double dz,
                         const Box box) {
    for (size_t z = box.z0; z < box.z1; ++z) {
        for (size_t y = box.y0; y < box.y1; ++y) {
            for (size_t x = box.x0; x < box.x1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

struct Domain {
    MPI_Comm comm;
    int rank, size, dims[3], coords[3], neighbors[6];
    size_t n[3], offset[3], pitch, plane;
    MPI_Datatype faces[3];

    Domain(MPI_Comm active, const int* decomposition, const size_t* global) {
        int periods[3] = {0, 0, 0};
        std::copy(decomposition, decomposition + 3, dims);
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int d = 0; d < 3; ++d) {
            n[d] = global[d] / dims[d] + (static_cast<size_t>(coords[d]) < global[d] % dims[d]);
            offset[d] = coords[d] * (global[d] / dims[d]) +
                        std::min(static_cast<size_t>(coords[d]), global[d] % dims[d]);
            MPI_Cart_shift(comm, d, 1, &neighbors[2*d], &neighbors[2*d+1]);
        }
        pitch = n[0] + 2;
        plane = pitch * (n[1] + 2);
        int sizes[3] = {static_cast<int>(n[2]+2), static_cast<int>(n[1]+2), static_cast<int>(n[0]+2)};
        int starts[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d) {
            int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
            subs[2-d] = 1;
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &faces[d]);
            MPI_Type_commit(&faces[d]);
        }
    }

    ~Domain() {
        for (auto& face : faces) MPI_Type_free(&face);
        MPI_Comm_free(&comm);
    }

    // Only face values are needed by the seven-point stencil (no edges/corners).
    void exchange(std::vector<double>& field, MPI_Request* requests) const {
        const size_t strides[3] = {1, pitch, plane};
        const size_t first = plane + pitch + 1;
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                int direction = 2*d + side;
                size_t send = first + (side ? n[d]-1 : 0) * strides[d];
                size_t recv = side ? send + strides[d] : send - strides[d];
                if (neighbors[direction] == MPI_PROC_NULL) {
                    const size_t ax = d == 0 ? 1 : n[0];
                    const size_t ay = d == 1 ? 1 : n[1];
                    const size_t az = d == 2 ? 1 : n[2];
                    for (size_t z = 0; z < az; ++z)
                        for (size_t y = 0; y < ay; ++y)
                            for (size_t x = 0; x < ax; ++x) {
                                size_t i = z*plane + y*pitch + x;
                                field[recv+i] = field[send+i];
                            }
                }
                MPI_Irecv(field.data()+recv, 1, faces[d], neighbors[direction],
                          direction^1, comm, &requests[direction]);
                MPI_Isend(field.data()+send, 1, faces[d], neighbors[direction],
                          direction, comm, &requests[6+direction]);
            }
        }
    }

    template<class Kernel>
    void stage(std::vector<double>& input, Kernel kernel) const {
        MPI_Request requests[12];
        exchange(input, requests);
        bool interior = n[0] > 2 && n[1] > 2 && n[2] > 2;
        if (interior) kernel(Box{2, n[0], 2, n[1], 2, n[2]});
        MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
        if (!interior) {
            kernel(Box{1, n[0]+1, 1, n[1]+1, 1, n[2]+1});
        } else {
            kernel(Box{1, n[0]+1, 1, n[1]+1, 1, 2});
            kernel(Box{1, n[0]+1, 1, n[1]+1, n[2], n[2]+1});
            kernel(Box{1, n[0]+1, 1, 2, 2, n[2]});
            kernel(Box{1, n[0]+1, n[1], n[1]+1, 2, n[2]});
            kernel(Box{1, 2, 2, n[1], 2, n[2]});
            kernel(Box{n[0], n[0]+1, 2, n[1], 2, n[2]});
        }
    }
};

// Gathering is only used for the optional, globally ordered result hash/output.
void printDistributedResults(const Domain& domain, const std::vector<double>& c,
                             const size_t* global) {
    std::vector<double> packed;
    packed.reserve(domain.n[0]*domain.n[1]*domain.n[2]);
    for (size_t z = 1; z <= domain.n[2]; ++z)
        for (size_t y = 1; y <= domain.n[1]; ++y) {
            auto row = c.begin() + z*domain.plane + y*domain.pitch + 1;
            packed.insert(packed.end(), row, row + domain.n[0]);
        }
    std::vector<double> full;
    if (domain.rank == 0) full.resize(global[0]*global[1]*global[2]);
    for (int rank = 0; rank < domain.size; ++rank) {
        if (domain.rank != 0 && domain.rank != rank) continue;
        int coords[3];
        MPI_Cart_coords(domain.comm, rank, 3, coords);
        size_t n[3], offset[3];
        for (int d = 0; d < 3; ++d) {
            n[d] = global[d]/domain.dims[d] + (static_cast<size_t>(coords[d]) < global[d]%domain.dims[d]);
            offset[d] = coords[d]*(global[d]/domain.dims[d]) +
                        std::min(static_cast<size_t>(coords[d]), global[d]%domain.dims[d]);
        }
        size_t count = n[0]*n[1]*n[2];
        if (domain.rank == 0 && rank != 0) packed.resize(count);
        for (size_t start = 0; rank != 0 && start < count;) {
            int chunk = static_cast<int>(std::min(count-start, static_cast<size_t>(INT_MAX)));
            if (domain.rank == 0)
                MPI_Recv(packed.data()+start, chunk, MPI_DOUBLE, rank, 0, domain.comm, MPI_STATUS_IGNORE);
            else
                MPI_Send(packed.data()+start, chunk, MPI_DOUBLE, 0, 0, domain.comm);
            start += chunk;
        }
        if (domain.rank == 0)
            for (size_t z = 0; z < n[2]; ++z)
                for (size_t y = 0; y < n[1]; ++y)
                    std::copy_n(packed.data()+(z*n[1]+y)*n[0], n[0],
                                full.data()+idx3(offset[0], offset[1]+y, offset[2]+z, global[0], global[1]));
    }
    if (domain.rank == 0) print_results(full, "Concentration");
}

bool validateResult(const std::vector<double>& c, const Domain& domain) {
    double local[3] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(), 0.0};
    for (size_t z = 1; z <= domain.n[2]; ++z)
        for (size_t y = 1; y <= domain.n[1]; ++y)
            for (size_t x = 1; x <= domain.n[0]; ++x) {
                double val = c[z*domain.plane+y*domain.pitch+x];
                local[0] = std::min(local[0], val);
                local[1] = std::min(local[1], -val);
                if (!std::isfinite(val)) local[2] = -1.0;
            }
    double global[3];
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MIN, domain.comm);
    if (domain.rank == 0) {
        if (global[2] < 0) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", global[0], -global[1]);
            if (global[0] < -10.0 || -global[1] > 10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return global[2] == 0 && global[0] >= -10.0 && -global[1] <= 10.0;
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
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize;
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > std::numeric_limits<size_t>::max()/ny ||
        nx*ny > std::numeric_limits<size_t>::max()/nz/sizeof(double)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }

    // Use as many ranks as possible without empty blocks, then minimize the
    // estimated halo surface. This also handles thin and anisotropic grids.
    const size_t global[3] = {nx, ny, nz};
    int decomposition[3] = {1, 1, 1};
    int activeSize = 1;
    double bestSurface = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= static_cast<int>(std::min(nx, static_cast<size_t>(worldSize))); ++px)
        for (int py = 1; py <= static_cast<int>(std::min(ny, static_cast<size_t>(worldSize/px))); ++py) {
            int pz = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize/px/py)));
            int used = px*py*pz;
            double surface = (px-1)*static_cast<double>(ny)*nz +
                             (py-1)*static_cast<double>(nx)*nz +
                             (pz-1)*static_cast<double>(nx)*ny;
            if (used > activeSize || (used == activeSize && surface < bestSurface)) {
                activeSize = used;
                bestSurface = surface;
                decomposition[0] = px; decomposition[1] = py; decomposition[2] = pz;
            }
        }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (worldRank >= activeSize) return 0;
    Domain domain(active, decomposition, global);
    MPI_Comm_free(&active);
    
    if (domain.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d x %d)\n", activeSize,
               decomposition[0], decomposition[1], decomposition[2]);
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
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    const size_t localSize = domain.plane * (domain.n[2]+2);
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);
    
    // Initialize concentration field
    if (domain.rank == 0) printf("Initializing concentration field...\n");
    for (size_t z = 1; z <= domain.n[2]; ++z)
        for (size_t y = 1; y <= domain.n[1]; ++y)
            for (size_t x = 1; x <= domain.n[0]; ++x) {
                const size_t id = idx3(domain.offset[0]+x-1, domain.offset[1]+y-1,
                                      domain.offset[2]+z-1, nx, ny);
                const double pseudo = (((id+1)*1299709)%gridSize)/static_cast<double>(gridSize);
                cold[z*domain.plane+y*domain.pitch+x] = -1.0+2.0*pseudo;
            }
    
    // Run simulation
    if (domain.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(domain.comm);
    double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        domain.stage(cold, [&](Box box) {
            computeChemicalPotential(cold, mu, domain.pitch, domain.n[1]+2, domain.n[2]+2,
                                     dx, dy, dz, gamma, e_AA, e_BB, e_AB, box);
        });
        
        // Update concentration
        domain.stage(mu, [&](Box box) {
            cahnHilliardUpdate(cnew, cold, mu, domain.pitch, domain.n[1]+2, domain.n[2]+2,
                              D, dt, dx, dy, dz, box);
        });
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    double elapsed = MPI_Wtime()-start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (domain.rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration*1000.0));

        // Use the slowest rank's elapsed time for global throughput.
        double cellUpdates = (double)gridSize * iterations;
        double mcups = duration > 0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        printDistributedResults(domain, cold, global);
    }
    
    // Validation
    if (validate) {
        if (domain.rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, domain);
        
        if (valid) {
            if (domain.rank == 0) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (domain.rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
