#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Choose the largest usable process grid, minimizing the area of exchanged
// faces. Unlike MPI_Dims_create, this also respects thin global dimensions.
int chooseGrid(int processes, const size_t global[3], int dims[3]) {
    for (int n = processes; n > 0; --n) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= n && static_cast<size_t>(x) <= global[0]; ++x) {
            if (n % x) continue;
            const int rest = n / x;
            for (int y = 1; y <= rest && static_cast<size_t>(y) <= global[1]; ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (static_cast<size_t>(z) > global[2]) continue;
                const double cost = static_cast<double>(x - 1) / global[0]
                                  + static_cast<double>(y - 1) / global[1]
                                  + static_cast<double>(z - 1) / global[2];
                if (cost < best) {
                    best = cost;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return n;
    }
    return 1;
}

void block(const size_t global[3], const int dims[3], const int coords[3],
           size_t local[3], size_t start[3]) {
    for (int d = 0; d < 3; ++d) {
        const size_t base = global[d] / dims[d], extra = global[d] % dims[d];
        local[d] = base + (static_cast<size_t>(coords[d]) < extra);
        start[d] = coords[d] * base + std::min(static_cast<size_t>(coords[d]), extra);
    }
}

struct Domain {
    MPI_Comm comm;
    size_t n[3], start[3], sx, plane;
    int neighbor[6];
    MPI_Datatype sendFace[6], receiveFace[6];
    MPI_Request requests[12];
    int requestCount = 0;

    Domain(MPI_Comm cart, const size_t global[3], const int dims[3]) : comm(cart) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        block(global, dims, coords, n, start);
        sx = n[0] + 2;
        plane = sx * (n[1] + 2);
        // MPI subarrays describe only face interiors: edge/corner halos are
        // unnecessary for a seven-point stencil.
        int sizes[3] = {static_cast<int>(n[2] + 2), static_cast<int>(n[1] + 2), static_cast<int>(n[0] + 2)};
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
            for (int side = 0; side < 2; ++side) {
                int face = 2*d + side;
                int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
                int origin[3] = {1, 1, 1};
                subs[2-d] = 1;
                origin[2-d] = side ? static_cast<int>(n[d]) : 1;
                MPI_Type_create_subarray(3, sizes, subs, origin, MPI_ORDER_C, MPI_DOUBLE, &sendFace[face]);
                MPI_Type_commit(&sendFace[face]);
                origin[2-d] = side ? static_cast<int>(n[d] + 1) : 0;
                MPI_Type_create_subarray(3, sizes, subs, origin, MPI_ORDER_C, MPI_DOUBLE, &receiveFace[face]);
                MPI_Type_commit(&receiveFace[face]);
            }
        }
    }

    ~Domain() {
        for (int f = 0; f < 6; ++f) {
            MPI_Type_free(&sendFace[f]);
            MPI_Type_free(&receiveFace[f]);
        }
    }

    size_t index(size_t x, size_t y, size_t z) const { return z * plane + y * sx + x; }

    void beginExchange(std::vector<double>& a) {
        requestCount = 0;
        for (int f = 0; f < 6; ++f) {
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Irecv(a.data(), 1, receiveFace[f], neighbor[f], f ^ 1, comm, &requests[requestCount++]);
        }
        for (int f = 0; f < 6; ++f) {
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Isend(a.data(), 1, sendFace[f], neighbor[f], f, comm, &requests[requestCount++]);
        }
        // Clamping is applied independently to concentration and potential.
        for (int d = 0; d < 3; ++d) {
            const size_t stride = d == 0 ? 1 : (d == 1 ? sx : plane);
            for (int side = 0; side < 2; ++side) {
                if (neighbor[2*d+side] != MPI_PROC_NULL) continue;
                size_t lo[3] = {1, 1, 1}, hi[3] = {n[0]+1, n[1]+1, n[2]+1};
                lo[d] = side ? n[d]+1 : 0;
                hi[d] = lo[d]+1;
                for (size_t z = lo[2]; z < hi[2]; ++z)
                    for (size_t y = lo[1]; y < hi[1]; ++y)
                        for (size_t x = lo[0]; x < hi[0]; ++x) {
                            const size_t i = index(x,y,z);
                            a[i] = a[side ? i-stride : i+stride];
                        }
            }
        }
    }

    void endExchange() { MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE); }

    template<class F>
    void region(size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1, F&& cell) const {
        for (size_t z = z0; z < z1; ++z)
            for (size_t y = y0; y < y1; ++y)
                for (size_t x = x0; x < x1; ++x) cell(index(x,y,z));
    }

    template<class F>
    void interior(F&& cell) const {
        if (n[0] > 2 && n[1] > 2 && n[2] > 2)
            region(2,n[0],2,n[1],2,n[2],cell);
    }

    template<class F>
    void boundary(F&& cell) const {
        if (n[0] <= 2 || n[1] <= 2 || n[2] <= 2) {
            region(1,n[0]+1,1,n[1]+1,1,n[2]+1,cell);
            return;
        }
        // Six disjoint slabs, avoiding a branch in the innermost loop.
        region(1,n[0]+1,1,n[1]+1,1,2,cell);
        region(1,n[0]+1,1,n[1]+1,n[2],n[2]+1,cell);
        region(1,n[0]+1,1,2,2,n[2],cell);
        region(1,n[0]+1,n[1],n[1]+1,2,n[2],cell);
        region(1,2,2,n[1],2,n[2],cell);
        region(n[0],n[0]+1,2,n[1],2,n[2],cell);
    }

    double laplacian(const std::vector<double>& a, size_t i) const {
        const double cxx = a[i+1] + a[i-1] - 2.0*a[i];
        const double cyy = a[i+sx] + a[i-sx] - 2.0*a[i];
        const double czz = a[i+plane] + a[i-plane] - 2.0*a[i];
        return cxx + cyy + czz;
    }
};

// Restore the original global x-fastest ordering only when results are requested.
// Row-wise transfers avoid MPI's int count/displacement limit for the full grid.
void printDistributedResults(const Domain& domain, const std::vector<double>& c,
                             const size_t global[3], const int dims[3], int rank, int ranks) {
    std::vector<double> full;
    if (rank == 0) full.resize(global[0]*global[1]*global[2]);
    for (int source = 0; source < ranks; ++source) {
        int coords[3];
        size_t n[3], start[3];
        MPI_Cart_coords(domain.comm, source, 3, coords);
        block(global, dims, coords, n, start);
        if (rank != 0 && rank != source) continue;
        for (size_t z = 0; z < n[2]; ++z) {
            for (size_t y = 0; y < n[1]; ++y) {
                if (rank == 0) {
                    double* row = full.data() + ((start[2]+z)*global[1]+start[1]+y)*global[0]+start[0];
                    if (source == 0)
                        std::copy_n(c.data()+domain.index(1,y+1,z+1), n[0], row);
                    else
                        MPI_Recv(row, static_cast<int>(n[0]), MPI_DOUBLE, source, 10, domain.comm, MPI_STATUS_IGNORE);
                } else {
                    MPI_Send(c.data()+domain.index(1,y+1,z+1), static_cast<int>(n[0]), MPI_DOUBLE, 0, 10, domain.comm);
                }
            }
        }
    }
    if (rank == 0) print_results(full, "Concentration");
}

bool validateResult(const Domain& domain, const std::vector<double>& c, int rank) {
    int bad = 0, globalBad;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    domain.region(1,domain.n[0]+1,1,domain.n[1]+1,1,domain.n[2]+1,[&](size_t i) {
        if (!std::isfinite(c[i])) bad = 1;
        low = std::min(low,c[i]);
        high = std::max(high,c[i]);
    });
    double globalLow, globalHigh;
    MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, domain.comm);
    MPI_Allreduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, domain.comm);
    MPI_Allreduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, domain.comm);
    if (rank == 0) {
        if (globalBad) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", globalLow, globalHigh);
            if (globalHigh > 10.0 || globalLow < -10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !globalBad && globalHigh <= 10.0 && globalLow >= -10.0;
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
    int worldRank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t global[3] = {nx, ny, nz};
    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > std::numeric_limits<size_t>::max()/ny ||
        nx*ny > std::numeric_limits<size_t>::max()/nz/sizeof(double)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    int dims[3];
    const int active = chooseGrid(processes, global, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    if (worldRank >= active) {
        MPI_Finalize();
        return 0;
    }
    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&workers);
    int rank;
    MPI_Comm_rank(cart, &rank);
    int result = 0;
    {
        Domain domain(cart, global, dims);
        const size_t gridSize = nx*ny*nz;
        const size_t localSize = domain.plane*(domain.n[2]+2);
        std::vector<double> cold(localSize), cnew(localSize), mu(localSize);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (size_t z = 1; z <= domain.n[2]; ++z)
            for (size_t y = 1; y <= domain.n[1]; ++y)
                for (size_t x = 1; x <= domain.n[0]; ++x) {
                    const size_t id = ((domain.start[2]+z-1)*ny+domain.start[1]+y-1)*nx+domain.start[0]+x-1;
                    const double pseudo = (((id+1)*1299709)%gridSize)/static_cast<double>(gridSize);
                    cold[domain.index(x,y,z)] = -1.0 + 2.0*pseudo;
                }

        // Preserve the original expressions and operation order (dx=dy=dz=1).
        const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
        const double gamma = 0.5, dt = 0.01, D = 1.0;
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            auto chemical = [&](size_t i) {
                const double cv = cold[i];
                mu[i] = 4.5*((cv+1.0)*e_AA+(cv-1.0)*e_BB-2.0*cv*e_AB)
                      + 3.0*cv + cv*cv*cv - gamma*domain.laplacian(cold,i);
            };
            domain.beginExchange(cold);
            domain.interior(chemical);
            domain.endExchange();
            domain.boundary(chemical);

            auto update = [&](size_t i) { cnew[i] = cold[i]+dt*D*domain.laplacian(mu,i); };
            domain.beginExchange(mu);
            domain.interior(update);
            domain.endExchange();
            domain.boundary(update);
            cold.swap(cnew);
        }
        const double elapsed = MPI_Wtime()-start;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration*1000.0));
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize)*iterations/duration/1e6);
        }
        if (printResults) printDistributedResults(domain, cold, global, dims, rank, active);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(domain, cold, rank);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return result;
}
