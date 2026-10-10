#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <vector>
// Use MPI's C interface; the deprecated C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

// Each rank owns a rectangular block with one face halo on each side.
// Edges and corners of the halo are unused by the seven-point stencil.
struct Domain {
    MPI_Comm comm;
    int rank, dims[3], coords[3], neighbor[6];
    size_t global[3], n[3], start[3], row, plane, volume;
    MPI_Datatype face[3];

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }

    static void partition(size_t extent, int parts, int coordinate,
                          size_t& length, size_t& offset) {
        const size_t base = extent / parts, remainder = extent % parts;
        length = base + (static_cast<size_t>(coordinate) < remainder);
        offset = base * coordinate + std::min(static_cast<size_t>(coordinate), remainder);
    }

    Domain(MPI_Comm active, const size_t* extents, const int* dimensions) {
        std::copy(extents, extents + 3, global);
        std::copy(dimensions, dimensions + 3, dims);
        int periodic[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periodic, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int d = 0; d < 3; ++d) {
            partition(global[d], dims[d], coords[d], n[d], start[d]);
            MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        }
        row = n[0] + 2;
        plane = row * (n[1] + 2);
        volume = n[0] * n[1] * n[2];
        // Subarray types transmit faces directly, without explicit packing buffers.
        int sizes[3] = {static_cast<int>(n[2]+2), static_cast<int>(n[1]+2),
                        static_cast<int>(n[0]+2)};
        int origin[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d) {
            int subsizes[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]),
                              static_cast<int>(n[0])};
            subsizes[2-d] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, origin, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[d]);
            MPI_Type_commit(&face[d]);
        }
    }

    void release() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Comm_free(&comm);
    }

    int exchange(std::vector<double>& field, MPI_Request* requests) const {
        int count = 0;
        // Clamp physical boundaries by replicating the owned boundary cells.
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                size_t send[3] = {1, 1, 1}, receive[3] = {1, 1, 1};
                send[d] = side ? n[d] : 1;
                receive[d] = side ? n[d]+1 : 0;
                const size_t source = index(send[0], send[1], send[2]);
                const size_t target = index(receive[0], receive[1], receive[2]);
                const int peer = neighbor[2*d+side];
                if (peer != MPI_PROC_NULL) {
                    MPI_Irecv(field.data()+target, 1, face[d], peer, 2*d+1-side,
                              comm, &requests[count++]);
                    MPI_Isend(field.data()+source, 1, face[d], peer, 2*d+side,
                              comm, &requests[count++]);
                } else if (d == 0) {
                    for (size_t z = 1; z <= n[2]; ++z)
                        for (size_t y = 1; y <= n[1]; ++y)
                            field[index(receive[0], y, z)] = field[index(send[0], y, z)];
                } else if (d == 1) {
                    for (size_t z = 1; z <= n[2]; ++z)
                        std::copy_n(field.data()+index(1, send[1], z), n[0],
                                    field.data()+index(1, receive[1], z));
                } else {
                    for (size_t y = 1; y <= n[1]; ++y)
                        std::copy_n(field.data()+index(1, y, send[2]), n[0],
                                    field.data()+index(1, y, receive[2]));
                }
            }
        }
        return count;
    }
};

// Select a process grid that respects thin dimensions and minimizes the
// largest block's surface area. Excess ranks wait outside the solver if the
// grid cannot accommodate them without empty blocks.
int processGrid(const size_t* extent, int processes, int* dims) {
    const size_t volume = extent[0]*extent[1]*extent[2];
    for (int count = static_cast<int>(std::min(volume, static_cast<size_t>(processes)));
         count > 0; --count) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= count && static_cast<size_t>(x) <= extent[0]; ++x) {
            if (count % x) continue;
            const int remaining = count/x;
            for (int y = 1; y <= remaining && static_cast<size_t>(y) <= extent[1]; ++y) {
                if (remaining % y) continue;
                const int z = remaining/y;
                if (static_cast<size_t>(z) > extent[2]) continue;
                const double lx = static_cast<double>((extent[0]+x-1)/x);
                const double ly = static_cast<double>((extent[1]+y-1)/y);
                const double lz = static_cast<double>((extent[2]+z-1)/z);
                const double surface = (x > 1 ? ly*lz : 0) +
                                       (y > 1 ? lx*lz : 0) + (z > 1 ? lx*ly : 0);
                // Penalize imbalance in uneven decompositions as well.
                const double score = surface * (lx*ly*lz*count/volume);
                if (score < best) {
                    best = score;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return count;
    }
    return 0;
}

// Unit spatial spacing and the physical constants match the original solver.
// Preserve the expression order, including the chemical potential terms.
template<bool chemical>
void computeBox(const Domain& d, const double* input, const double* cold,
                double* __restrict__ output,
                size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
    constexpr double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
    for (size_t z = z0; z < z1; ++z) {
        for (size_t y = y0; y < y1; ++y) {
            const size_t base = d.index(0, y, z);
            for (size_t x = x0; x < x1; ++x) {
                const size_t i = base + x;
                const double cv = input[i];
                const double cxx = input[i+1] + input[i-1] - 2.0*cv;
                const double cyy = input[i+d.row] + input[i-d.row] - 2.0*cv;
                const double czz = input[i+d.plane] + input[i-d.plane] - 2.0*cv;
                const double laplacian = cxx + cyy + czz;
                if constexpr (chemical) {
                    output[i] = 4.5*((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
                                + 3.0*cv + cv*cv*cv - 0.5*laplacian;
                } else {
                    output[i] = cold[i] + 0.01*laplacian;
                }
            }
        }
    }
}

template<bool chemical>
void stencil(const Domain& d, std::vector<double>& input,
             const std::vector<double>& cold, std::vector<double>& output) {
    MPI_Request requests[12];
    const int count = d.exchange(input, requests);
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    computeBox<chemical>(d, input.data(), cold.data(), output.data(),
                         2, nx, 2, ny, 2, nz);
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    // Disjoint slabs cover the boundary, including blocks only one cell thick.
    auto box = [&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
        computeBox<chemical>(d, input.data(), cold.data(), output.data(),
                             x0, x1, y0, y1, z0, z1);
    };
    box(1, nx+1, 1, ny+1, 1, 2);
    if (nz > 1) box(1, nx+1, 1, ny+1, nz, nz+1);
    box(1, nx+1, 1, 2, 2, nz);
    if (ny > 1) box(1, nx+1, ny, ny+1, 2, nz);
    box(1, 2, 2, ny, 2, nz);
    if (nx > 1) box(nx, nx+1, 2, ny, 2, nz);
}

void initializeConcentration(const Domain& d, std::vector<double>& c) {
    const size_t volume = d.global[0]*d.global[1]*d.global[2];
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const size_t id = ((d.start[2]+z-1)*d.global[1]+d.start[1]+y-1)
                                  *d.global[0]+d.start[0]+x-1;
                const double pseudo = (((id+1)*1299709)%volume)/static_cast<double>(volume);
                c[d.index(x,y,z)] = -1.0+2.0*pseudo;
            }
}

bool validateResult(const Domain& d, const std::vector<double>& c) {
    int finite = 1, allFinite;
    double extrema[2] = {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const double value = c[d.index(x,y,z)];
                if (!std::isfinite(value)) finite = 0;
                extrema[0] = std::min(extrema[0], value);
                extrema[1] = std::min(extrema[1], -value);
            }
    double global[2];
    MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, d.comm);
    MPI_Allreduce(extrema, global, 2, MPI_DOUBLE, MPI_MIN, d.comm);
    const bool valid = allFinite && global[0] >= -10.0 && -global[1] <= 10.0;
    if (d.rank == 0) {
        if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", global[0], -global[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
    }
    return valid;
}

// Gathering is only needed for -r: reconstruct global x-fastest order to
// retain the original Kahan sum, samples, and order-dependent result hash.
void printResults(const Domain& d, const std::vector<double>& c) {
    std::vector<double> packed(d.volume);
    size_t offset = 0;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y) {
            std::copy_n(c.data()+d.index(1,y,z), d.n[0], packed.data()+offset);
            offset += d.n[0];
        }
    int processes;
    MPI_Comm_size(d.comm, &processes);
    if (d.rank != 0) {
        for (size_t i = 0; i < packed.size();) {
            const int count = static_cast<int>(std::min(packed.size()-i, static_cast<size_t>(INT_MAX)));
            MPI_Send(packed.data()+i, count, MPI_DOUBLE, 0, 20, d.comm);
            i += count;
        }
        return;
    }
    std::vector<double> global(d.global[0]*d.global[1]*d.global[2]);
    for (int rank = 0; rank < processes; ++rank) {
        int coords[3];
        size_t n[3], start[3];
        MPI_Cart_coords(d.comm, rank, 3, coords);
        for (int axis = 0; axis < 3; ++axis)
            Domain::partition(d.global[axis], d.dims[axis], coords[axis], n[axis], start[axis]);
        if (rank != 0) {
            packed.resize(n[0]*n[1]*n[2]);
            for (size_t i = 0; i < packed.size();) {
                const int count = static_cast<int>(std::min(packed.size()-i, static_cast<size_t>(INT_MAX)));
                MPI_Recv(packed.data()+i, count, MPI_DOUBLE, rank, 20, d.comm, MPI_STATUS_IGNORE);
                i += count;
            }
        }
        offset = 0;
        for (size_t z = 0; z < n[2]; ++z)
            for (size_t y = 0; y < n[1]; ++y) {
                const size_t target = ((start[2]+z)*d.global[1]+start[1]+y)*d.global[0]+start[0];
                std::copy_n(packed.data()+offset, n[0], global.data()+target);
                offset += n[0];
            }
    }
    print_results(global, "Concentration");
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t extent[3] = {64, 0, 0};
    int iterations = 20;
    bool validate = false, results = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        int axis = -1;
        if (strcmp(argv[i], "-x") == 0) axis = 0;
        else if (strcmp(argv[i], "-y") == 0) axis = 1;
        else if (strcmp(argv[i], "-z") == 0) axis = 2;
        if ((axis >= 0 || strcmp(argv[i], "-i") == 0) && i+1 < argc) {
            char* end;
            const long value = std::strtol(argv[++i], &end, 10);
            if (*end != '\0' || end == argv[i] || value < 0 || value > INT_MAX ||
                (axis == 0 && value == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                status = 1;
                break;
            }
            if (axis >= 0) extent[axis] = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }
    if (!extent[1]) extent[1] = extent[0];
    if (!extent[2]) extent[2] = extent[0];
    const size_t limit = std::numeric_limits<size_t>::max()/sizeof(double);
    if (extent[0] > INT_MAX-2 || extent[1] > INT_MAX-2 || extent[2] > INT_MAX-2 ||
        extent[0] > limit/extent[1] || extent[0]*extent[1] > limit/extent[2]) {
        if (rank == 0) fprintf(stderr, "Grid dimensions are too large\n");
        status = 1;
    }
    if (status) {
        MPI_Finalize();
        return status;
    }
    int dims[3];
    const int activeCount = processGrid(extent, processes, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeCount ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank < activeCount) {
        try {
            Domain d(active, extent, dims);
            if (rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", extent[0], extent[1], extent[2]);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("MPI process grid: %d x %d x %d (%d active ranks)\n",
                       dims[0], dims[1], dims[2], activeCount);
                printf("Initializing concentration field...\n");
            }
            if (d.row > limit/(d.n[1]+2) || d.plane > limit/(d.n[2]+2))
                throw std::bad_alloc();
            const size_t storage = d.plane*(d.n[2]+2);
            std::vector<double> cold(storage), cnew(storage), mu(storage);
            initializeConcentration(d, cold);
            if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(d.comm);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                stencil<true>(d, cold, cold, mu);
                stencil<false>(d, mu, cold, cnew);
                cold.swap(cnew);
            }
            const double elapsed = MPI_Wtime()-start;
            double duration;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(duration*1000.0));
                const double updates = static_cast<double>(extent[0])*extent[1]*extent[2]*iterations;
                printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates/duration/1e6 : 0.0);
            }
            if (results) printResults(d, cold);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateResult(d, cold);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
            d.release();
            MPI_Comm_free(&active);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Idle ranks participate in exit status propagation, not solver collectives.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
