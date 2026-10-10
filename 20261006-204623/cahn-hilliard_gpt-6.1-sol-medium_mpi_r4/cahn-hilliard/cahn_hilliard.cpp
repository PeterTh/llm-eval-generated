#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Choose a Cartesian decomposition that fits even thin or very small grids.
// Maximize participating ranks, then minimize the total internal face area.
void chooseDecomposition(const size_t global[3], int ranks, int dims[3]) {
    int used = 0;
    long double best = std::numeric_limits<long double>::max();
    for (int x = 1; x <= int(std::min(global[0], size_t(ranks))); ++x) {
        for (int y = 1; y <= int(std::min(global[1], size_t(ranks / x))); ++y) {
            const int z = int(std::min(global[2], size_t(ranks / x / y)));
            const int n = x * y * z;
            const long double area = (x - 1) * static_cast<long double>(global[1]) * global[2]
                                   + (y - 1) * static_cast<long double>(global[0]) * global[2]
                                   + (z - 1) * static_cast<long double>(global[0]) * global[1];
            if (n > used || (n == used && area < best)) {
                used = n;
                best = area;
                dims[0] = x; dims[1] = y; dims[2] = z;
            }
        }
    }
}

struct Domain {
    MPI_Comm comm;
    int n[3], neighbor[6];
    size_t start[3], sx, sy, cells;
    MPI_Datatype face[3];

    Domain(MPI_Comm cart, const size_t global[3], const int dims[3]) : comm(cart) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int d = 0; d < 3; ++d) {
            const size_t base = global[d] / dims[d], extra = global[d] % dims[d];
            n[d] = int(base + (size_t(coords[d]) < extra));
            start[d] = size_t(coords[d]) * base + std::min(size_t(coords[d]), extra);
            MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        }
        sx = size_t(n[0]) + 2;
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
        if (sx > limit / (size_t(n[1]) + 2))
            throw std::length_error("Local grid is too large");
        sy = sx * (size_t(n[1]) + 2);
        if (sy > limit / (size_t(n[2]) + 2))
            throw std::length_error("Local grid is too large");
        cells = sy * (size_t(n[2]) + 2);
        const int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        const int origin[3] = {0, 0, 0};
        for (int d = 0; d < 3; ++d) {
            int subsizes[3] = {n[2], n[1], n[0]};
            subsizes[2-d] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, origin, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[d]);
            MPI_Type_commit(&face[d]);
        }
    }

    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
    }

    size_t index(int x, int y, int z) const {
        return size_t(z) * sy + size_t(y) * sx + size_t(x);
    }

    // Faces exclude edges/corners: neither stencil reads diagonal neighbors.
    int exchange(std::vector<double>& field, MPI_Request requests[12]) const {
        int count = 0;
        const size_t stride[3] = {1, sx, sy};
        double* first = field.data() + index(1, 1, 1);
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                const int peer = neighbor[2*d+side];
                if (peer == MPI_PROC_NULL) continue;
                double* send = first + (side ? size_t(n[d]-1) * stride[d] : 0);
                double* recv = side ? first + size_t(n[d]) * stride[d] : first - stride[d];
                MPI_Irecv(recv, 1, face[d], peer, 2*d+1-side, comm, &requests[count++]);
                MPI_Isend(send, 1, face[d], peer, 2*d+side, comm, &requests[count++]);
            }
        }
        return count;
    }

    // The arithmetic ordering matches the original serial stencil. Grid spacing
    // is exactly one in all dimensions in this benchmark.
    template<bool chemical>
    void box(const std::vector<double>& input, const std::vector<double>& cold,
             std::vector<double>& output, const int lo[3], const int hi[3]) const {
        const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
        for (int z = lo[2]; z <= hi[2]; ++z) {
            const size_t zn = (z == 1 && neighbor[4] == MPI_PROC_NULL) ? 0 : sy;
            const size_t zp = (z == n[2] && neighbor[5] == MPI_PROC_NULL) ? 0 : sy;
            for (int y = lo[1]; y <= hi[1]; ++y) {
                const size_t yn = (y == 1 && neighbor[2] == MPI_PROC_NULL) ? 0 : sx;
                const size_t yp = (y == n[1] && neighbor[3] == MPI_PROC_NULL) ? 0 : sx;
                int first = lo[0], last = hi[0];
                if (first > last) continue;
                const double* src = input.data();
                const double* old = cold.data();
                double* dst = output.data();
                const auto cell = [&](size_t i, size_t xn, size_t xp) {
                    const double cv = src[i];
                    const double cxx = src[i+xp] + src[i-xn] - 2.0 * cv;
                    const double cyy = src[i+yp] + src[i-yn] - 2.0 * cv;
                    const double czz = src[i+zp] + src[i-zn] - 2.0 * cv;
                    const double laplacian = cxx + cyy + czz;
                    if constexpr (chemical) {
                        dst[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                               + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
                    } else {
                        dst[i] = old[i] + 0.01 * 1.0 * laplacian;
                    }
                };
                // Peel physical x boundaries so the main row loop uses fixed
                // offsets and can be vectorized without gathers or branches.
                if (first == 1 && neighbor[0] == MPI_PROC_NULL) {
                    cell(index(first,y,z), 0,
                         (first == n[0] && neighbor[1] == MPI_PROC_NULL) ? 0 : 1);
                    ++first;
                }
                const bool right = last == n[0] && neighbor[1] == MPI_PROC_NULL && first <= last;
                if (right) --last;
                const size_t begin = index(first,y,z);
                const size_t end = begin + size_t(std::max(0, last-first+1));
                for (size_t i = begin; i < end; ++i) cell(i, 1, 1);
                if (right) cell(index(last+1,y,z), 1, 0);
            }
        }
    }

    template<bool chemical>
    void step(std::vector<double>& input, const std::vector<double>& cold,
              std::vector<double>& output) const {
        MPI_Request requests[12];
        const int count = exchange(input, requests);
        int lo[3], hi[3];
        bool interior = true;
        for (int d = 0; d < 3; ++d) {
            lo[d] = neighbor[2*d] == MPI_PROC_NULL ? 1 : 2;
            hi[d] = n[d] - (neighbor[2*d+1] != MPI_PROC_NULL);
            interior = interior && lo[d] <= hi[d];
        }
        if (interior) box<chemical>(input, cold, output, lo, hi);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        int a[3] = {1, 1, 1}, b[3] = {n[0], n[1], n[2]};
        if (!interior) {
            box<chemical>(input, cold, output, a, b);
            return;
        }
        // Six disjoint boxes cover the remaining boundary shell.
        for (int d = 0; d < 3; ++d) {
            const int end = b[d];
            b[d] = lo[d] - 1;
            box<chemical>(input, cold, output, a, b);
            b[d] = end;
            a[d] = hi[d] + 1;
            box<chemical>(input, cold, output, a, b);
            a[d] = lo[d]; b[d] = hi[d];
        }
    }
};

// Gather only on request, reconstructing the serial x-fastest ordering. Chunked
// transfers avoid MPI's int count limit, and workers store only their own block.
void printDistributedResults(const Domain& domain, const std::vector<double>& c,
                             const size_t global[3], const int dims[3]) {
    int rank, ranks;
    MPI_Comm_rank(domain.comm, &rank);
    MPI_Comm_size(domain.comm, &ranks);
    std::vector<double> packed(size_t(domain.n[0]) * domain.n[1] * domain.n[2]);
    size_t offset = 0;
    for (int z = 1; z <= domain.n[2]; ++z)
        for (int y = 1; y <= domain.n[1]; ++y) {
            std::copy_n(c.data() + domain.index(1,y,z), domain.n[0], packed.data() + offset);
            offset += domain.n[0];
        }
    if (rank != 0) {
        for (size_t i = 0; i < packed.size();) {
            const int count = int(std::min(packed.size()-i, size_t(INT_MAX)));
            MPI_Send(packed.data()+i, count, MPI_DOUBLE, 0, 0, domain.comm);
            i += count;
        }
        return;
    }
    std::vector<double> result(global[0] * global[1] * global[2]);
    for (int r = 0; r < ranks; ++r) {
        int coords[3];
        size_t start[3], n[3];
        MPI_Cart_coords(domain.comm, r, 3, coords);
        for (int d = 0; d < 3; ++d) {
            const size_t base = global[d] / dims[d], extra = global[d] % dims[d];
            n[d] = base + (size_t(coords[d]) < extra);
            start[d] = size_t(coords[d]) * base + std::min(size_t(coords[d]), extra);
        }
        if (r != 0) {
            packed.resize(n[0]*n[1]*n[2]);
            for (size_t i = 0; i < packed.size();) {
                const int count = int(std::min(packed.size()-i, size_t(INT_MAX)));
                MPI_Recv(packed.data()+i, count, MPI_DOUBLE, r, 0, domain.comm, MPI_STATUS_IGNORE);
                i += count;
            }
        }
        offset = 0;
        for (size_t z = 0; z < n[2]; ++z)
            for (size_t y = 0; y < n[1]; ++y) {
                const size_t target = ((start[2]+z)*global[1]+start[1]+y)*global[0]+start[0];
                std::copy_n(packed.data()+offset, n[0], result.data()+target);
                offset += n[0];
            }
    }
    print_results(result, "Concentration");
}

bool validateResult(const Domain& domain, const std::vector<double>& c) {
    int bad = 0, globalBad, rank;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -minimum, globalMin, globalMax;
    for (int z = 1; z <= domain.n[2]; ++z)
        for (int y = 1; y <= domain.n[1]; ++y)
            for (int x = 1; x <= domain.n[0]; ++x) {
                const double v = c[domain.index(x,y,z)];
                bad |= !std::isfinite(v);
                minimum = std::min(minimum, v);
                maximum = std::max(maximum, v);
            }
    MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, domain.comm);
    MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, domain.comm);
    MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, domain.comm);
    MPI_Comm_rank(domain.comm, &rank);
    const bool valid = !globalBad && globalMin >= -10.0 && globalMax <= 10.0;
    if (rank == 0) {
        if (globalBad) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
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
    size_t global[3] = {64, 0, 0};
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i+1 < argc) global[0] = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i+1 < argc) global[1] = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i+1 < argc) global[2] = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (worldRank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    size_t volume = 1;
    for (const size_t n : global) {
        if (n == 0 || n > size_t(INT_MAX-2) || volume > std::numeric_limits<size_t>::max()/n) {
            if (worldRank == 0) fprintf(stderr, "Invalid or oversized grid dimensions\n");
            MPI_Finalize();
            return 1;
        }
        volume *= n;
    }
    int dims[3];
    chooseDecomposition(global, worldSize, dims);
    const int activeRanks = dims[0]*dims[1]*dims[2];
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &active);
    int status = 0;
    if (active != MPI_COMM_NULL) {
        const int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(active, 3, dims, periods, 0, &cart);
        try {
            Domain domain(cart, global, dims);
            std::vector<double> cold(domain.cells), cnew(domain.cells), mu(domain.cells);
            if (worldRank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("Initializing concentration field...\n");
            }
            for (int z = 1; z <= domain.n[2]; ++z)
                for (int y = 1; y <= domain.n[1]; ++y)
                    for (int x = 1; x <= domain.n[0]; ++x) {
                        const size_t id = ((domain.start[2]+z-1)*global[1]+domain.start[1]+y-1)*global[0]+domain.start[0]+x-1;
                        const double pseudo = (((id+1)*1299709) % volume) / static_cast<double>(volume);
                        cold[domain.index(x,y,z)] = -1.0 + 2.0*pseudo;
                    }
            if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            const double begin = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                domain.step<true>(cold, cold, mu);
                domain.step<false>(mu, cold, cnew);
                cold.swap(cnew);
            }
            const double elapsed = MPI_Wtime()-begin;
            double duration;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (worldRank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(duration*1000.0));
                printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/duration/1e6);
            }
            if (printResults) printDistributedResults(domain, cold, global, dims);
            if (validate) {
                if (worldRank == 0) printf("Validating result...\n");
                status = validateResult(domain, cold) ? 0 : 1;
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
