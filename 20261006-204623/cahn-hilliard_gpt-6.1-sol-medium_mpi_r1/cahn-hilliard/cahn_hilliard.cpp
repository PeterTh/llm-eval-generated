#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Every rank owns a rectangular block with one ghost cell on each face.
// Edges and corners of the ghost region are unused by the seven-point stencil.
struct Domain {
    MPI_Comm comm;
    int rank, dims[3], coords[3], neighbor[6];
    size_t n[3], origin[3], sy, sz, storage;
    MPI_Datatype face[3];

    Domain(MPI_Comm communicator, const size_t global[3]) : comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        int periods[3];
        MPI_Cart_get(comm, 3, dims, periods, coords);
        for (int a = 0; a < 3; ++a) {
            const size_t base = global[a] / dims[a], extra = global[a] % dims[a];
            n[a] = base + (static_cast<size_t>(coords[a]) < extra);
            origin[a] = coords[a] * base + std::min(static_cast<size_t>(coords[a]), extra);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
            if (n[a] > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
                if (rank == 0) fprintf(stderr, "Local grid dimension exceeds MPI datatype limits\n");
                MPI_Abort(comm, 1);
            }
        }
        sy = n[0] + 2;
        sz = sy * (n[1] + 2);
        storage = sz * (n[2] + 2);
        int sizes[3] = {static_cast<int>(n[2]+2), static_cast<int>(n[1]+2), static_cast<int>(n[0]+2)};
        int starts[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) {
            int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
            subs[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
    }

    size_t index(size_t x, size_t y, size_t z) const { return z*sz + y*sy + x; }

    void exchange(std::vector<double>& field, MPI_Request requests[12]) const {
        const size_t stride[3] = {1, sy, sz};
        const size_t first = index(1, 1, 1);
        for (int a = 0; a < 3; ++a) {
            const size_t last = first + (n[a]-1)*stride[a];
            // Opposite faces use matching tags, including when a block is one cell wide.
            MPI_Irecv(field.data()+first-stride[a], 1, face[a], neighbor[2*a], 2*a+1, comm, &requests[4*a]);
            MPI_Irecv(field.data()+last+stride[a], 1, face[a], neighbor[2*a+1], 2*a, comm, &requests[4*a+1]);
            MPI_Isend(field.data()+first, 1, face[a], neighbor[2*a], 2*a, comm, &requests[4*a+2]);
            MPI_Isend(field.data()+last, 1, face[a], neighbor[2*a+1], 2*a+1, comm, &requests[4*a+3]);
        }
        // Replicate the boundary cell on physical faces, as in the original solver.
        if (neighbor[0] == MPI_PROC_NULL || neighbor[1] == MPI_PROC_NULL)
            for (size_t z = 1; z <= n[2]; ++z)
                for (size_t y = 1; y <= n[1]; ++y) {
                    if (neighbor[0] == MPI_PROC_NULL) field[index(0,y,z)] = field[index(1,y,z)];
                    if (neighbor[1] == MPI_PROC_NULL) field[index(n[0]+1,y,z)] = field[index(n[0],y,z)];
                }
        if (neighbor[2] == MPI_PROC_NULL || neighbor[3] == MPI_PROC_NULL)
            for (size_t z = 1; z <= n[2]; ++z)
                for (size_t x = 1; x <= n[0]; ++x) {
                    if (neighbor[2] == MPI_PROC_NULL) field[index(x,0,z)] = field[index(x,1,z)];
                    if (neighbor[3] == MPI_PROC_NULL) field[index(x,n[1]+1,z)] = field[index(x,n[1],z)];
                }
        if (neighbor[4] == MPI_PROC_NULL || neighbor[5] == MPI_PROC_NULL)
            for (size_t y = 1; y <= n[1]; ++y)
                for (size_t x = 1; x <= n[0]; ++x) {
                    if (neighbor[4] == MPI_PROC_NULL) field[index(x,y,0)] = field[index(x,y,1)];
                    if (neighbor[5] == MPI_PROC_NULL) field[index(x,y,n[2]+1)] = field[index(x,y,n[2])];
                }
    }

    ~Domain() { for (auto& type : face) MPI_Type_free(&type); }
};

inline double laplacian(const double* field, size_t i, size_t sy, size_t sz) {
    const double cv = field[i];
    const double cxx = field[i+1] + field[i-1] - 2.0*cv;
    const double cyy = field[i+sy] + field[i-sy] - 2.0*cv;
    const double czz = field[i+sz] + field[i-sz] - 2.0*cv;
    return cxx + cyy + czz;
}

// Contiguous row kernels keep the inner loop free of boundary branches.
template<bool chemical>
void compute(const Domain& d, const std::vector<double>& cold,
             const std::vector<double>& input, std::vector<double>& output, bool interior) {
    const double* c = cold.data();
    const double* f = input.data();
    double* out = output.data();
    const auto row = [&](size_t y, size_t z, size_t begin, size_t end) {
        const size_t offset = d.index(0,y,z);
        for (size_t x = begin; x < end; ++x) {
            const size_t i = offset+x;
            if constexpr (chemical) {
                const double cv = c[i];
                constexpr double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
                out[i] = 4.5*((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
                         + 3.0*cv + cv*cv*cv - 0.5*laplacian(f,i,d.sy,d.sz);
            } else {
                out[i] = c[i] + 0.01*laplacian(f,i,d.sy,d.sz);
            }
        }
    };
    if (interior) {
        for (size_t z = 2; z < d.n[2]; ++z)
            for (size_t y = 2; y < d.n[1]; ++y)
                row(y,z,2,d.n[0]);
    } else {
        for (size_t z = 1; z <= d.n[2]; ++z)
            for (size_t y = 1; y <= d.n[1]; ++y) {
                if (z == 1 || z == d.n[2] || y == 1 || y == d.n[1] || d.n[0] <= 2)
                    row(y,z,1,d.n[0]+1);
                else {
                    row(y,z,1,2);
                    row(y,z,d.n[0],d.n[0]+1);
                }
            }
    }
}

// Gather only for requested external validation. Chunked transfers avoid the
// int count/displacement limit of MPI_Gatherv and preserve global linear order.
std::vector<double> gather(const Domain& d, const std::vector<double>& c, const size_t global[3]) {
    int ranks;
    MPI_Comm_size(d.comm, &ranks);
    const size_t localSize = d.n[0]*d.n[1]*d.n[2];
    std::vector<double> packed(localSize);
    size_t offset = 0;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y) {
            std::copy_n(c.data()+d.index(1,y,z), d.n[0], packed.data()+offset);
            offset += d.n[0];
        }
    constexpr size_t chunk = 1u << 26;
    std::vector<double> result;
    if (d.rank != 0) {
        for (size_t pos = 0; pos < localSize; pos += chunk)
            MPI_Send(packed.data()+pos, static_cast<int>(std::min(chunk,localSize-pos)), MPI_DOUBLE, 0, 100, d.comm);
    } else {
        result.resize(global[0]*global[1]*global[2]);
        for (int rank = 0; rank < ranks; ++rank) {
            int coords[3];
            MPI_Cart_coords(d.comm, rank, 3, coords);
            size_t n[3], origin[3];
            for (int a = 0; a < 3; ++a) {
                size_t base = global[a]/d.dims[a], extra = global[a]%d.dims[a];
                n[a] = base + (static_cast<size_t>(coords[a]) < extra);
                origin[a] = coords[a]*base + std::min(static_cast<size_t>(coords[a]),extra);
            }
            const size_t count = n[0]*n[1]*n[2];
            if (rank != 0) {
                packed.resize(count);
                for (size_t pos = 0; pos < count; pos += chunk)
                    MPI_Recv(packed.data()+pos, static_cast<int>(std::min(chunk,count-pos)), MPI_DOUBLE, rank, 100, d.comm, MPI_STATUS_IGNORE);
            }
            offset = 0;
            for (size_t z = 0; z < n[2]; ++z)
                for (size_t y = 0; y < n[1]; ++y) {
                    size_t dest = ((origin[2]+z)*global[1]+origin[1]+y)*global[0]+origin[0];
                    std::copy_n(packed.data()+offset,n[0],result.data()+dest);
                    offset += n[0];
                }
        }
    }
    return result;
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i+1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i+1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i+1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t global[3] = {nx,ny,nz};
    const size_t limit = std::numeric_limits<size_t>::max()/sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit/ny || nx*ny > limit/nz) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }

    // Choose the largest feasible rank grid, then minimize the total halo area.
    // This also supports thin grids and more MPI ranks than cells.
    int dims[3] = {1,1,1}, active = 1;
    double best = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= static_cast<int>(std::min(nx,static_cast<size_t>(worldSize))); ++x)
        for (int y = 1; y <= static_cast<int>(std::min(ny,static_cast<size_t>(worldSize/x))); ++y) {
            int z = static_cast<int>(std::min(nz,static_cast<size_t>(worldSize/(x*y))));
            int used = x*y*z;
            double area = (x > 1 ? double(x-1)/nx : 0.0)
                        + (y > 1 ? double(y-1)/ny : 0.0)
                        + (z > 1 ? double(z-1)/nz : 0.0);
            if (used > active || (used == active && area < best)) {
                active = used; best = area;
                dims[0] = x; dims[1] = y; dims[2] = z;
            }
        }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    int status = 0;
    if (workers != MPI_COMM_NULL) {
        int periods[3] = {0,0,0};
        MPI_Comm cart;
        MPI_Cart_create(workers,3,dims,periods,0,&cart);
        {
            Domain d(cart,global);
            if (d.rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", nx,ny,nz);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("MPI ranks: %d (%d x %d x %d)\n", active,dims[0],dims[1],dims[2]);
                printf("Initializing concentration field...\n");
            }
            std::vector<double> cold(d.storage), cnew(d.storage), mu(d.storage);
            const size_t volume = nx*ny*nz;
            for (size_t z = 1; z <= d.n[2]; ++z)
                for (size_t y = 1; y <= d.n[1]; ++y)
                    for (size_t x = 1; x <= d.n[0]; ++x) {
                        const size_t id = ((d.origin[2]+z-1)*ny + d.origin[1]+y-1)*nx + d.origin[0]+x-1;
                        const double pseudo = (((id+1)*1299709)%volume)/static_cast<double>(volume);
                        cold[d.index(x,y,z)] = -1.0 + 2.0*pseudo;
                    }
            if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                MPI_Request requests[12];
                d.exchange(cold,requests);
                compute<true>(d,cold,cold,mu,true);
                MPI_Waitall(12,requests,MPI_STATUSES_IGNORE);
                compute<true>(d,cold,cold,mu,false);
                d.exchange(mu,requests);
                compute<false>(d,cold,mu,cnew,true);
                MPI_Waitall(12,requests,MPI_STATUSES_IGNORE);
                compute<false>(d,cold,mu,cnew,false);
                cold.swap(cnew);
            }
            double elapsed = MPI_Wtime()-start, duration;
            MPI_Reduce(&elapsed,&duration,1,MPI_DOUBLE,MPI_MAX,0,cart);
            if (d.rank == 0) {
                printf("Computation time: %.3f ms\n", duration*1000.0);
                printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/duration/1e6);
            }
            if (printResults) {
                auto result = gather(d,cold,global);
                if (d.rank == 0) print_results(result,"Concentration");
            }
            if (validate) {
                double localMin = std::numeric_limits<double>::infinity();
                double localMax = -std::numeric_limits<double>::infinity();
                int invalid = 0, anyInvalid;
                for (size_t z = 1; z <= d.n[2]; ++z)
                    for (size_t y = 1; y <= d.n[1]; ++y)
                        for (size_t x = 1; x <= d.n[0]; ++x) {
                            double value = cold[d.index(x,y,z)];
                            invalid |= !std::isfinite(value);
                            localMin = std::min(localMin,value);
                            localMax = std::max(localMax,value);
                        }
                double minVal,maxVal;
                MPI_Allreduce(&invalid,&anyInvalid,1,MPI_INT,MPI_MAX,cart);
                MPI_Allreduce(&localMin,&minVal,1,MPI_DOUBLE,MPI_MIN,cart);
                MPI_Allreduce(&localMax,&maxVal,1,MPI_DOUBLE,MPI_MAX,cart);
                status = anyInvalid || minVal < -10.0 || maxVal > 10.0;
                if (d.rank == 0) {
                    printf("Validating result...\n");
                    if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n",minVal,maxVal);
                        if (status) printf("Validation failed: values out of expected range\n");
                    }
                    printf("Validation: %s\n",status ? "FAILED" : "PASSED");
                }
            }
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
