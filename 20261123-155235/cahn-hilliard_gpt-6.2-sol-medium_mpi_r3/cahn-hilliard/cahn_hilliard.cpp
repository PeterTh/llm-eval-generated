#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

using Dimensions = std::array<int, 3>; // x, y, z
using Strides = std::array<size_t, 3>;

size_t offset(const Dimensions& point, const Strides& stride) {
    return point[0] * stride[0] + point[1] * stride[1] + point[2] * stride[2];
}

std::vector<int> divisors(int n) {
    std::vector<int> factors;
    for (int i = 1; i <= n / i; ++i) {
        if (n % i == 0) {
            factors.push_back(i);
            if (i != n / i) factors.push_back(n / i);
        }
    }
    return factors;
}

Dimensions processGrid(const Dimensions& global, int ranks) {
    // Use as many nonempty blocks as possible. For a given rank count, favor
    // small worst-case blocks and low face traffic. Prefer x-contiguous faces
    // when decompositions have the same block size and communication volume.
    for (int count = ranks; count > 0; --count) {
        Dimensions best{};
        double bestCost = std::numeric_limits<double>::infinity();
        size_t bestStridedFaces = std::numeric_limits<size_t>::max();
        for (int px : divisors(count)) {
            if (px > global[0]) continue;
            for (int py : divisors(count / px)) {
                const int pz = count / px / py;
                if (py > global[1] || pz > global[2]) continue;
                const Dimensions grid{px, py, pz};
                Strides extent;
                for (int axis = 0; axis < 3; ++axis) {
                    extent[axis] = 1 + (global[axis] - 1) / grid[axis];
                }
                const size_t xFace = extent[1] * extent[2];
                const size_t yFace = extent[0] * extent[2];
                const size_t zFace = extent[0] * extent[1];
                const size_t stridedFaces = std::min(px - 1, 2) * xFace +
                                            std::min(py - 1, 2) * yFace;
                const size_t faces = stridedFaces + std::min(pz - 1, 2) * zFace;
                const double cost = static_cast<double>(extent[0] * xFace) + faces;
                if (cost < bestCost || (cost == bestCost && stridedFaces < bestStridedFaces)) {
                    best = grid;
                    bestCost = cost;
                    bestStridedFaces = stridedFaces;
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

struct Block {
    Dimensions origin;
    Dimensions size;
};

Block blockForRank(MPI_Comm cart, int rank, const Dimensions& global, const Dimensions& grid) {
    Dimensions coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    Block block;
    for (int axis = 0; axis < 3; ++axis) {
        const int coordinate = coords[2 - axis]; // MPI's last coordinate varies fastest.
        const int base = global[axis] / grid[axis];
        const int remainder = global[axis] % grid[axis];
        block.origin[axis] = coordinate * base + std::min(coordinate, remainder);
        block.size[axis] = base + (coordinate < remainder);
    }
    return block;
}

// Describe an x-fast rectangular section, including the gaps between rows and planes.
MPI_Datatype blockType(const Dimensions& size, const Strides& stride) {
    MPI_Datatype type;
    MPI_Type_contiguous(size[0], MPI_DOUBLE, &type);
    for (int axis = 1; axis < 3; ++axis) {
        if (size[axis] == 1) continue;
        MPI_Datatype next;
        MPI_Type_create_hvector(size[axis], 1, stride[axis] * sizeof(double), type, &next);
        MPI_Type_free(&type);
        type = next;
    }
    MPI_Type_commit(&type);
    return type;
}

struct Box {
    std::array<size_t, 3> begin;
    std::array<size_t, 3> end;

    bool empty() const {
        return begin[0] == end[0] || begin[1] == end[1] || begin[2] == end[2];
    }
};

struct Face {
    int neighbor;
    size_t owned;
    size_t halo;
};

// Each field uses stable storage, allowing persistent point-to-point requests
// to exchange ghost faces directly between padded blocks.
class HaloExchange {
public:
    HaloExchange(MPI_Comm cart, const Dimensions& size, const Strides& stride,
                 const std::array<double*, 3>& fields)
        : interior{{1, 1, 1}, {static_cast<size_t>(size[0]) + 1,
                               static_cast<size_t>(size[1]) + 1,
                               static_cast<size_t>(size[2]) + 1}},
          size(size), stride(stride) {
        for (int axis = 2; axis >= 0; --axis) {
            Face& low = faces[2 * axis];
            Face& high = faces[2 * axis + 1];
            MPI_Cart_shift(cart, 2 - axis, 1, &low.neighbor, &high.neighbor);
            const size_t first = offset({1, 1, 1}, stride);
            low.owned = first;
            low.halo = first - stride[axis];
            high.owned = first + (size[axis] - 1) * stride[axis];
            high.halo = high.owned + stride[axis];

            if (low.neighbor == MPI_PROC_NULL && high.neighbor == MPI_PROC_NULL) continue;
            Dimensions faceSize = size;
            faceSize[axis] = 1;
            MPI_Datatype type = blockType(faceSize, stride);
            for (int side = 0; side < 2; ++side) {
                const int direction = 2 * axis + side;
                const Face& face = faces[direction];
                if (face.neighbor == MPI_PROC_NULL) continue;
                for (size_t i = 0; i < fields.size(); ++i) {
                    MPI_Recv_init(fields[i] + face.halo, 1, type, face.neighbor,
                                  direction ^ 1, cart, &requests[i][requestCount]);
                    MPI_Send_init(fields[i] + face.owned, 1, type, face.neighbor,
                                  direction, cart, &requests[i][requestCount + 1]);
                }
                requestCount += 2;

                // Peel off disjoint strips that depend on a remote face.
                if (!interior.empty()) {
                    Box strip = interior;
                    if (side == 0) {
                        strip.end[axis] = ++interior.begin[axis];
                    } else {
                        strip.begin[axis] = --interior.end[axis];
                    }
                    boundary[boundaryCount++] = strip;
                }
            }
            MPI_Type_free(&type);
        }
    }

    ~HaloExchange() {
        for (auto& fieldRequests : requests) {
            for (int i = 0; i < requestCount; ++i) MPI_Request_free(&fieldRequests[i]);
        }
    }

    void start(int field, double* data) {
        if (requestCount != 0) MPI_Startall(requestCount, requests[field].data());
        for (int axis = 0; axis < 3; ++axis) {
            for (int side = 0; side < 2; ++side) {
                const Face& face = faces[2 * axis + side];
                if (face.neighbor == MPI_PROC_NULL) copyFace(data, face, axis);
            }
        }
    }

    void finish(int field) {
        if (requestCount != 0) MPI_Waitall(requestCount, requests[field].data(), MPI_STATUSES_IGNORE);
    }

    Box interior;
    std::array<Box, 6> boundary;
    int boundaryCount = 0;

private:
    void copyFace(double* data, const Face& face, int axis) const {
        // Clamp the global boundary by copying its owned face into the adjacent ghost cells.
        const double* source = data + face.owned;
        double* destination = data + face.halo;
        if (axis == 0) {
            for (int z = 0; z < size[2]; ++z) {
                for (int y = 0; y < size[1]; ++y) {
                    const size_t i = z * stride[2] + y * stride[1];
                    destination[i] = source[i];
                }
            }
        } else {
            const int other = axis == 1 ? 2 : 1;
            for (int j = 0; j < size[other]; ++j) {
                const size_t i = j * stride[other];
                std::copy_n(source + i, size[0], destination + i);
            }
        }
    }

    Dimensions size;
    Strides stride;
    std::array<Face, 6> faces;
    std::array<std::array<MPI_Request, 12>, 3> requests;
    int requestCount = 0;
};

void initializeConcentration(double* data, const Block& block, const Strides& stride,
                             const Strides& globalStride, size_t volume) {
    for (int z = 0; z < block.size[2]; ++z) {
        for (int y = 0; y < block.size[1]; ++y) {
            const size_t local = offset({1, y + 1, z + 1}, stride);
            const size_t global = offset({block.origin[0], block.origin[1] + y, block.origin[2] + z},
                                         globalStride);
            for (int x = 0; x < block.size[0]; ++x) {
                const size_t linearId = global + x;
                const double pseudo = ((linearId + 1) * 1299709 % volume) / static_cast<double>(volume);
                data[local + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

template <bool chemicalPotential>
void stencil(const Box& box, const Strides& stride, const double* __restrict__ input,
             const double* __restrict__ cold, double* __restrict__ output) {
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double dt = 0.01;
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t row = z * stride[2] + y * stride[1];
            for (size_t x = box.begin[0]; x < box.end[0]; ++x) {
                const size_t i = row + x;
                const double center = input[i];
                const double xx = input[i + 1] + input[i - 1] - 2.0 * center;
                const double yy = input[i + stride[1]] + input[i - stride[1]] - 2.0 * center;
                const double zz = input[i + stride[2]] + input[i - stride[2]] - 2.0 * center;
                const double laplacian = xx + yy + zz;
                if constexpr (chemicalPotential) {
                    output[i] = 4.5 * ((center + 1.0) * eAA + (center - 1.0) * eBB - 2.0 * center * eAB)
                                + 3.0 * center + center * center * center - gamma * laplacian;
                } else {
                    output[i] = cold[i] + dt * laplacian;
                }
            }
        }
    }
}

template <bool chemicalPotential>
void advance(HaloExchange& halo, int field, const Strides& stride, double* input,
             const double* cold, double* output) {
    halo.start(field, input);
    if (!halo.interior.empty()) stencil<chemicalPotential>(halo.interior, stride, input, cold, output);
    halo.finish(field);
    for (int b = 0; b < halo.boundaryCount; ++b) {
        stencil<chemicalPotential>(halo.boundary[b], stride, input, cold, output);
    }
}

void reportResults(MPI_Comm cart, int rank, int ranks, const Dimensions& global,
                   const Dimensions& grid, const Block& block, const Strides& stride,
                   const Strides& globalStride, size_t volume, const double* data) {
    constexpr int resultTag = 6;
    if (rank != 0) {
        MPI_Datatype type = blockType(block.size, stride);
        MPI_Send(data + offset({1, 1, 1}, stride), 1, type, 0, resultTag, cart);
        MPI_Type_free(&type);
        return;
    }

    // The existing report computes an ordered hash and sum over the global field.
    std::vector<double> result(volume);
    std::vector<MPI_Request> pending(ranks - 1);
    for (int peer = 1; peer < ranks; ++peer) {
        const Block remote = blockForRank(cart, peer, global, grid);
        MPI_Datatype type = blockType(remote.size, globalStride);
        MPI_Irecv(result.data() + offset(remote.origin, globalStride), 1, type,
                  peer, resultTag, cart, &pending[peer - 1]);
        MPI_Type_free(&type);
    }
    for (int z = 0; z < block.size[2]; ++z) {
        for (int y = 0; y < block.size[1]; ++y) {
            const size_t source = offset({1, y + 1, z + 1}, stride);
            const size_t destination = offset({block.origin[0], block.origin[1] + y, block.origin[2] + z},
                                              globalStride);
            std::copy_n(data + source, block.size[0], result.data() + destination);
        }
    }
    MPI_Waitall(ranks - 1, pending.data(), MPI_STATUSES_IGNORE);
    print_results(result, "Concentration");
}

bool validateResult(MPI_Comm cart, int rank, const Block& block, const Strides& stride,
                    const double* data) {
    const double inf = std::numeric_limits<double>::infinity();
    std::array<double, 3> local{-inf, -inf, 0.0};
    for (int z = 0; z < block.size[2]; ++z) {
        for (int y = 0; y < block.size[1]; ++y) {
            const size_t row = offset({1, y + 1, z + 1}, stride);
            for (int x = 0; x < block.size[0]; ++x) {
                const double value = data[row + x];
                if (std::isfinite(value)) {
                    local[0] = std::max(local[0], -value);
                    local[1] = std::max(local[1], value);
                } else {
                    local[2] = 1.0;
                }
            }
        }
    }

    std::array<double, 3> global;
    MPI_Allreduce(local.data(), global.data(), 3, MPI_DOUBLE, MPI_MAX, cart);
    const double minValue = -global[0];
    const double maxValue = global[1];
    if (global[2] != 0.0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const bool valid = minValue >= -10.0 && maxValue <= 10.0;
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
        if (!valid) printf("Validation failed: values out of expected range\n");
    }
    return valid;
}

void printTiming(double elapsed, size_t volume, int iterations) {
    printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
    const double mcups = elapsed > 0.0 ? static_cast<double>(volume) * iterations / elapsed / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
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

struct Options {
    Dimensions size;
    int iterations;
    int validate;
    int report;
    int status;
};

Options parseOptions(int argc, char** argv) {
    Options options{{64, 0, 0}, 20, 0, 0, 0};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.size[0] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.size[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.size[2] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.report = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            options.status = 1;
            return options;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            options.status = 2;
            return options;
        }
    }
    if (options.size[1] == 0) options.size[1] = options.size[0];
    if (options.size[2] == 0) options.size[2] = options.size[0];

    size_t volume = 1;
    constexpr size_t maxElements = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
    for (int n : options.size) {
        if (n < 0 || (n != 0 && volume > maxElements / static_cast<size_t>(n))) {
            printf("Invalid grid size\n");
            options.status = 2;
            break;
        }
        volume *= n;
    }
    return options;
}

int run(MPI_Comm cart, const Options& options, const Dimensions& grid, size_t volume,
        const Strides& globalStride) {
    int rank, ranks;
    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &ranks);
    const Block block = blockForRank(cart, rank, options.size, grid);
    const size_t pitch = static_cast<size_t>(block.size[0]) + 2;
    const size_t rows = static_cast<size_t>(block.size[1]) + 2;
    const size_t planes = static_cast<size_t>(block.size[2]) + 2;
    constexpr size_t maxElements = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
    if (pitch > maxElements / rows / planes) throw std::bad_array_new_length();
    const Strides stride{1, pitch, pitch * rows};
    const size_t storage = stride[2] * planes;
    auto first = std::make_unique_for_overwrite<double[]>(storage);
    auto second = std::make_unique_for_overwrite<double[]>(storage);
    auto mu = std::make_unique_for_overwrite<double[]>(storage);
    const std::array<double*, 3> fields{first.get(), second.get(), mu.get()};
    HaloExchange halo(cart, block.size, stride, fields);

    initializeConcentration(first.get(), block, stride, globalStride, volume);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();

    int current = 0;
    for (int t = 0; t < options.iterations; ++t) {
        const int next = current ^ 1;
        advance<true>(halo, current, stride, fields[current], nullptr, mu.get());
        advance<false>(halo, 2, stride, mu.get(), fields[current], fields[next]);
        current = next;
    }

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) printTiming(maxElapsed, volume, options.iterations);

    if (options.report) {
        reportResults(cart, rank, ranks, options.size, grid, block, stride, globalStride,
                      volume, fields[current]);
    }
    if (options.validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cart, rank, block, stride, fields[current]);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int benchmark(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::array<int, 7> arguments;
    if (rank == 0) {
        const Options options = parseOptions(argc, argv);
        arguments = {options.size[0], options.size[1], options.size[2], options.iterations,
                     options.validate, options.report, options.status};
    }
    MPI_Bcast(arguments.data(), arguments.size(), MPI_INT, 0, MPI_COMM_WORLD);
    const Options options{{arguments[0], arguments[1], arguments[2]}, arguments[3],
                          arguments[4], arguments[5], arguments[6]};
    if (options.status != 0) return options.status - 1;

    const Dimensions& global = options.size;
    const Strides globalStride{1, static_cast<size_t>(global[0]),
                               static_cast<size_t>(global[0]) * global[1]};
    const size_t volume = globalStride[2] * global[2];

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Time steps: %d\n", options.iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    if (volume == 0) {
        if (rank == 0) {
            printf("Running Cahn-Hilliard simulation...\n");
            printTiming(0.0, 0, options.iterations);
            if (options.report) print_results(std::vector<double>{}, "Concentration");
            if (options.validate) {
                printf("Validating result...\n");
                printf("Validation failed: empty concentration field\n");
                printf("Validation: FAILED\n");
            }
        }
        return options.validate ? 1 : 0;
    }

    const Dimensions grid = processGrid(global, static_cast<int>(std::min<size_t>(ranks, volume)));
    const Dimensions cartSize{grid[2], grid[1], grid[0]};
    const Dimensions periods{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, cartSize.data(), periods.data(), 0, &cart);
    int result = 0;
    if (cart != MPI_COMM_NULL) {
        result = run(cart, options, grid, volume, globalStride);
        MPI_Comm_free(&cart);
    }
    // Carry the validation status to any ranks that did not get a grid block.
    if (options.validate && grid[0] * grid[1] * grid[2] != ranks) {
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = benchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
