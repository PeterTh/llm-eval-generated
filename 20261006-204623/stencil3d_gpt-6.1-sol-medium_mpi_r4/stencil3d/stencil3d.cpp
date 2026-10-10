#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

using Real = double;

// Choose a feasible process grid with the smallest estimated face traffic.
// Searching factors also handles thin grids and more ranks than grid cells.
std::array<int, 3> processGrid(int ranks, const std::array<int, 3>& sizes) {
    for (int n = ranks; n > 0; --n) {
        double best = std::numeric_limits<double>::infinity();
        std::array<int, 3> result{};
        for (int x = 1; x <= std::min(n, sizes[0]); ++x) {
            if (n % x) continue;
            const int remaining = n / x;
            for (int y = 1; y <= std::min(remaining, sizes[1]); ++y) {
                if (remaining % y) continue;
                const int z = remaining / y;
                if (z > sizes[2]) continue;
                const double cost = double(x - 1) / sizes[0] +
                                    double(y - 1) / sizes[1] +
                                    double(z - 1) / sizes[2];
                if (cost < best) {
                    best = cost;
                    result = {x, y, z};
                }
            }
        }
        if (result[0]) return result;
    }
    return {1, 1, 1};
}

// The local storage has one ghost cell on every face, with X contiguous.
MPI_Datatype blockType(const std::array<int, 3>& storage,
                       const std::array<int, 3>& extent,
                       const std::array<int, 3>& offset) {
    int full[3] = {storage[2], storage[1], storage[0]};
    int sub[3] = {extent[2], extent[1], extent[0]};
    int start[3] = {offset[2], offset[1], offset[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, full, sub, start, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Keep the original addition order and division for reproducible results.
void updateRow(const Real* input, Real* output, size_t row, int begin, int end,
               size_t pitch, size_t plane) {
    for (int x = begin; x < end; ++x) {
        const size_t i = row + x;
        output[i] = (input[i] + input[i - 1] + input[i + 1] +
                     input[i - pitch] + input[i + pitch] +
                     input[i - plane] + input[i + plane]) / 7.0;
    }
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    std::array<int, 3> sizes = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false, invalid = false, help = false;
    for (int i = 1; i < argc; ++i) {
        int* value = nullptr;
        if (!strcmp(argv[i], "-x")) value = &sizes[0];
        else if (!strcmp(argv[i], "-y")) value = &sizes[1];
        else if (!strcmp(argv[i], "-z")) value = &sizes[2];
        else if (!strcmp(argv[i], "-i")) value = &iterations;
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else invalid = true;
        if (value) {
            if (i + 1 == argc) { invalid = true; break; }
            char* end = nullptr;
            const long long parsed = std::strtoll(argv[++i], &end, 10);
            if (!*argv[i] || *end || parsed < 0 ||
                parsed > std::numeric_limits<int>::max() - 2) invalid = true;
            else *value = static_cast<int>(parsed);
        }
    }
    if (!sizes[1]) sizes[1] = sizes[0];
    if (!sizes[2]) sizes[2] = sizes[0];
    size_t gridSize = 1;
    for (int size : sizes) {
        if (size <= 0 || gridSize > std::numeric_limits<size_t>::max() / sizeof(Real) /
                                       static_cast<size_t>(std::max(size, 1))) {
            invalid = true;
            break;
        }
        gridSize *= static_cast<size_t>(size);
    }
    if (help || invalid) {
        if (worldRank == 0) {
            if (invalid) printf("Invalid option or grid size\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return invalid ? 1 : 0;
    }

    const auto dims = processGrid(worldSize, sizes);
    const int activeSize = dims[0] * dims[1] * dims[2];
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &active);
    // Spare ranks still participate in final status propagation and finalization.
    int status = 0;
    if (active != MPI_COMM_NULL) {
        MPI_Comm cart;
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims.data(), periods, 0, &cart);
        int rank;
        MPI_Comm_rank(cart, &rank);
        std::array<int, 3> coords, local, offset, storage, begin, end;
        MPI_Cart_coords(cart, rank, 3, coords.data());
        for (int d = 0; d < 3; ++d) {
            const int base = sizes[d] / dims[d], extra = sizes[d] % dims[d];
            local[d] = base + (coords[d] < extra);
            offset[d] = coords[d] * base + std::min(coords[d], extra);
            storage[d] = local[d] + 2;
            begin[d] = offset[d] == 0 ? 2 : 1;
            end[d] = local[d] + 1 - (offset[d] + local[d] == sizes[d]);
        }
        size_t localSize = 1;
        for (int size : storage) {
            if (localSize > std::numeric_limits<size_t>::max() / sizeof(Real) /
                                static_cast<size_t>(size)) {
                fprintf(stderr, "Rank %d: local grid size overflows address space\n", worldRank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            localSize *= static_cast<size_t>(size);
        }
        const size_t pitch = storage[0], plane = pitch * storage[1];
        std::vector<Real> grid1, grid2;
        try {
            grid1.resize(localSize);
            grid2.resize(localSize);
        } catch (...) {
            fprintf(stderr, "Rank %d: unable to allocate local grids\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int z = 1; z <= local[2]; ++z) {
            for (int y = 1; y <= local[1]; ++y) {
                const size_t globalRow = (size_t(offset[2] + z - 1) * sizes[1] +
                                          offset[1] + y - 1) * sizes[0] + offset[0];
                const size_t row = size_t(z) * plane + size_t(y) * pitch;
                for (int x = 1; x <= local[0]; ++x)
                    grid1[row + x] = grid2[row + x] = Real((globalRow + x - 1) % 19);
            }
        }
        std::array<int, 6> neighbors;
        std::array<MPI_Datatype, 6> sendTypes{}, receiveTypes{};
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart, d, 1, &neighbors[2*d], &neighbors[2*d+1]);
            for (int side = 0; side < 2; ++side) {
                const int face = 2*d + side;
                if (neighbors[face] == MPI_PROC_NULL) continue;
                auto extent = local;
                extent[d] = 1;
                std::array<int, 3> start = {1, 1, 1};
                start[d] = side ? local[d] : 1;
                sendTypes[face] = blockType(storage, extent, start);
                start[d] = side ? local[d] + 1 : 0;
                receiveTypes[face] = blockType(storage, extent, start);
            }
        }
        if (rank == 0) {
            printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", sizes[0], sizes[1], sizes[2]);
            printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (%d x %d x %d)\n", activeSize, dims[0], dims[1], dims[2]);
            printf("Initializing grid...\nRunning stencil computation...\n");
        }
        Real* input = grid1.data();
        Real* output = grid2.data();
        std::array<int, 3> coreBegin, coreEnd;
        for (int d = 0; d < 3; ++d) {
            coreBegin[d] = std::max(begin[d], 2);
            coreEnd[d] = std::max(coreBegin[d], std::min(end[d], local[d]));
        }
        MPI_Barrier(cart);
        const double startTime = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Request requests[12];
            int count = 0;
            for (int f = 0; f < 6; ++f) {
                if (neighbors[f] == MPI_PROC_NULL) continue;
                MPI_Irecv(input, 1, receiveTypes[f], neighbors[f], f ^ 1,
                          cart, &requests[count++]);
            }
            for (int f = 0; f < 6; ++f) {
                if (neighbors[f] == MPI_PROC_NULL) continue;
                MPI_Isend(input, 1, sendTypes[f], neighbors[f], f,
                          cart, &requests[count++]);
            }
            for (int z = coreBegin[2]; z < coreEnd[2]; ++z)
                for (int y = coreBegin[1]; y < coreEnd[1]; ++y)
                    updateRow(input, output, size_t(z)*plane + size_t(y)*pitch,
                              coreBegin[0], coreEnd[0], pitch, plane);
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            for (int z = begin[2]; z < end[2]; ++z) {
                for (int y = begin[1]; y < end[1]; ++y) {
                    const size_t row = size_t(z)*plane + size_t(y)*pitch;
                    if (z < coreBegin[2] || z >= coreEnd[2] ||
                        y < coreBegin[1] || y >= coreEnd[1]) {
                        updateRow(input, output, row, begin[0], end[0], pitch, plane);
                    } else {
                        updateRow(input, output, row, begin[0], std::min(end[0], coreBegin[0]), pitch, plane);
                        updateRow(input, output, row, coreEnd[0], end[0], pitch, plane);
                    }
                }
            }
            std::swap(input, output);
        }
        const double elapsed = MPI_Wtime() - startTime;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            const double updates = double(std::max(sizes[0]-2, 0)) *
                std::max(sizes[1]-2, 0) * std::max(sizes[2]-2, 0) * iterations;
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
        }
        if (printResults) {
            // Gather only on request; receive blocks directly into global order.
            MPI_Datatype owned = blockType(storage, local, {1, 1, 1});
            if (rank == 0) {
                std::vector<Real> global;
                try { global.resize(gridSize); }
                catch (...) { MPI_Abort(MPI_COMM_WORLD, 1); }
                for (int z = 1; z <= local[2]; ++z)
                    for (int y = 1; y <= local[1]; ++y)
                        std::copy_n(input + size_t(z)*plane + size_t(y)*pitch + 1,
                                    local[0], global.data() +
                                    (size_t(offset[2]+z-1)*sizes[1]+offset[1]+y-1)*sizes[0]+offset[0]);
                for (int source = 1; source < activeSize; ++source) {
                    std::array<int, 3> c, length, origin;
                    MPI_Cart_coords(cart, source, 3, c.data());
                    for (int d = 0; d < 3; ++d) {
                        const int base = sizes[d] / dims[d], extra = sizes[d] % dims[d];
                        length[d] = base + (c[d] < extra);
                        origin[d] = c[d]*base + std::min(c[d], extra);
                    }
                    MPI_Datatype destination = blockType(sizes, length, origin);
                    MPI_Recv(global.data(), 1, destination, source, 6, cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&destination);
                }
                print_results(global, "Grid");
            } else MPI_Send(input, 1, owned, 0, 6, cart);
            MPI_Type_free(&owned);
        }
        if (validate) {
            Real minimum = std::numeric_limits<Real>::infinity();
            Real maximum = -minimum;
            int valid = 1;
            for (int z = 1; z <= local[2]; ++z)
                for (int y = 1; y <= local[1]; ++y)
                    for (int x = 1; x <= local[0]; ++x) {
                        const Real value = input[size_t(z)*plane + size_t(y)*pitch + x];
                        if (!std::isfinite(value)) valid = 0;
                        minimum = std::min(minimum, value);
                        maximum = std::max(maximum, value);
                    }
            Real globalMin, globalMax;
            int globalValid;
            MPI_Reduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
            MPI_Reduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            MPI_Reduce(&valid, &globalValid, 1, MPI_INT, MPI_MIN, 0, cart);
            if (rank == 0) {
                printf("Validating result...\n");
                if (!globalValid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                    if (globalMax > 1e6 || globalMin < -1e6) {
                        printf("Validation failed: values out of expected range\n");
                        globalValid = 0;
                    }
                }
                printf("Validation: %s\n", globalValid ? "PASSED" : "FAILED");
                status = globalValid ? 0 : 1;
            }
        }
        for (int f = 0; f < 6; ++f) {
            if (neighbors[f] == MPI_PROC_NULL) continue;
            MPI_Type_free(&sendTypes[f]);
            MPI_Type_free(&receiveTypes[f]);
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
