#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;
using Extent = std::array<int, 3>;  // x, y, z

struct Box {
    Extent begin;
    Extent end;

    Extent size() const {
        return {end[0] - begin[0], end[1] - begin[1], end[2] - begin[2]};
    }
};

size_t index_of(int x, int y, int z, const Extent& shape) {
    return (static_cast<size_t>(z) * shape[1] + y) * shape[0] + x;
}

size_t volume(const Extent& shape) {
    return static_cast<size_t>(shape[0]) * shape[1] * shape[2];
}

// Compare the largest block and its halo for each feasible process grid.
// Permit a small number of idle ranks when the rank count would force a thin grid.
Extent process_grid(const Extent& work, int ranks) {
    int maximum = 0;
    for (int px = 1; px <= std::min(work[0], ranks); ++px) {
        for (int py = 1; py <= std::min(work[1], ranks / px); ++py) {
            const int pz = std::min(work[2], ranks / px / py);
            maximum = std::max(maximum, px * py * pz);
        }
    }

    Extent best{1, 1, 1};
    double bestCost = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= std::min(work[0], ranks); ++px) {
        const double x = 1 + (work[0] - 1) / px;
        for (int py = 1; py <= std::min(work[1], ranks / px); ++py) {
            const int pz = std::min(work[2], ranks / px / py);
            if (px * py * pz < maximum - maximum / 8) continue;

            const double y = 1 + (work[1] - 1) / py;
            const double z = 1 + (work[2] - 1) / pz;
            const double halo = std::min(px - 1, 2) * y * z +
                                std::min(py - 1, 2) * x * z +
                                std::min(pz - 1, 2) * x * y;
            const double cost = x * y * z + 4 * halo;
            if (cost < bestCost) {
                bestCost = cost;
                best = {px, py, pz};
            }
        }
    }
    return best;
}

struct Domain {
    Extent origin;  // Global coordinate of local index (0, 0, 0).
    Extent shape;
    Box owned;      // Includes any physical boundary cells on this rank.
    Box update;     // Interior cells to update.
};

Domain make_domain(const Extent& work, const Extent& dims, const Extent& coords,
                   bool hasInterior) {
    Domain domain{};
    for (int axis = 0; axis < 3; ++axis) {
        const int base = work[axis] / dims[axis];
        const int extra = work[axis] % dims[axis];
        const int count = base + (coords[axis] < extra);
        domain.origin[axis] = coords[axis] * base + std::min(coords[axis], extra);
        domain.shape[axis] = count + (hasInterior ? 2 : 0);
        domain.owned.begin[axis] = hasInterior && coords[axis] > 0 ? 1 : 0;
        domain.owned.end[axis] = domain.shape[axis] -
            (hasInterior && coords[axis] + 1 < dims[axis] ? 1 : 0);
        domain.update.begin[axis] = hasInterior ? 1 : 0;
        domain.update.end[axis] = count + domain.update.begin[axis];
    }
    return domain;
}

void initialize_grids(Real* first, Real* second, const Domain& domain,
                      const Extent& global, int iterations, bool hasInterior) {
    const auto& [begin, end] = domain.owned;
    for (int z = begin[2]; z < end[2]; ++z) {
        for (int y = begin[1]; y < end[1]; ++y) {
            const size_t local = index_of(begin[0], y, z, domain.shape);
            const size_t offset = index_of(domain.origin[0],
                domain.origin[1] + y, domain.origin[2] + z, global);
            for (int x = begin[0]; x < end[0]; ++x) {
                const size_t index = local + (x - begin[0]);
                first[index] = (offset + x) % 19;
                // Keep physical boundaries fixed in both buffers. The unused
                // second buffer is zero if a negative odd iteration count selects it.
                if (iterations < 0) {
                    second[index] = 0;
                } else if (!hasInterior || x == 0 || x == domain.shape[0] - 1 ||
                           y == 0 || y == domain.shape[1] - 1 ||
                           z == 0 || z == domain.shape[2] - 1) {
                    second[index] = first[index];
                }
            }
        }
    }
}

void stencil(const Real* __restrict input, Real* __restrict output,
             const Extent& shape, const Box& box) {
    const auto& [begin, end] = box;
    if (begin[0] == end[0] || begin[1] == end[1] || begin[2] == end[2]) return;
    const size_t row = shape[0];
    const size_t plane = row * shape[1];
    for (int z = begin[2]; z < end[2]; ++z) {
        for (int y = begin[1]; y < end[1]; ++y) {
            const size_t start = index_of(begin[0], y, z, shape);
            const size_t stop = start + (end[0] - begin[0]);
            for (size_t i = start; i < stop; ++i) {
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - row] + input[i + row] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

MPI_Datatype subarray(const Extent& shape, const Box& box) {
    const Extent size = box.size();
    // MPI_ORDER_FORTRAN gives the first dimension (x) unit stride.
    MPI_Datatype type;
    MPI_Type_create_subarray(3, shape.data(), size.data(), box.begin.data(),
                             MPI_ORDER_FORTRAN, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct HaloExchange {
    std::array<std::array<MPI_Request, 12>, 2> requests;
    std::array<Box, 6> boundary;
    Box core;
    int count = 0;
    int faces = 0;

    HaloExchange(const Domain& domain, MPI_Comm cart, const std::array<Real*, 2>& grids)
        : core(domain.update) {
        for (int axis = 0; axis < 3; ++axis) {
            int neighbors[2];
            MPI_Cart_shift(cart, axis, 1, &neighbors[0], &neighbors[1]);
            for (int side = 0; side < 2; ++side) {
                if (neighbors[side] == MPI_PROC_NULL) continue;

                const int face = 2 * axis + side;
                Box send = domain.update;
                const int edge = side == 0 ? send.begin[axis] : send.end[axis] - 1;
                send.begin[axis] = edge;
                send.end[axis] = edge + 1;
                Box receive = send;
                const int direction = side == 0 ? -1 : 1;
                receive.begin[axis] += direction;
                receive.end[axis] += direction;

                MPI_Datatype sendType = subarray(domain.shape, send);
                MPI_Datatype receiveType = subarray(domain.shape, receive);
                for (int buffer = 0; buffer < 2; ++buffer) {
                    MPI_Recv_init(grids[buffer], 1, receiveType, neighbors[side],
                                  face ^ 1, cart, &requests[buffer][count]);
                    MPI_Send_init(grids[buffer], 1, sendType, neighbors[side],
                                  face, cart, &requests[buffer][count + 1]);
                }
                count += 2;
                MPI_Type_free(&receiveType);
                MPI_Type_free(&sendType);

                // Peel each face from the remaining box so thin blocks, edges
                // and corners are updated exactly once after communication.
                if (core.begin[axis] < core.end[axis]) {
                    Box skin = core;
                    if (side == 0) {
                        skin.end[axis] = ++core.begin[axis];
                    } else {
                        skin.begin[axis] = --core.end[axis];
                    }
                    boundary[faces++] = skin;
                }
            }
        }
    }

    ~HaloExchange() {
        for (auto& buffer : requests) {
            for (int i = 0; i < count; ++i) MPI_Request_free(&buffer[i]);
        }
    }
};

void print_global_results(const Real* grid, const Domain& domain, const Extent& global,
                          const Extent& work, const Extent& dims, bool hasInterior,
                          int rank, MPI_Comm cart) {
    constexpr int tag = 6;
    if (rank != 0) {
        MPI_Datatype owned = subarray(domain.shape, domain.owned);
        MPI_Send(grid, 1, owned, 0, tag, cart);
        MPI_Type_free(&owned);
        return;
    }

    // The common reporting code needs global x-fastest order for the hash,
    // compensated sum and samples. Only allocate this grid when -r is used.
    std::vector<Real> result(volume(global));
    const auto& [begin, end] = domain.owned;
    for (int z = begin[2]; z < end[2]; ++z) {
        for (int y = begin[1]; y < end[1]; ++y) {
            const size_t src = index_of(begin[0], y, z, domain.shape);
            const size_t dst = index_of(domain.origin[0] + begin[0],
                domain.origin[1] + y, domain.origin[2] + z, global);
            std::copy_n(grid + src, end[0] - begin[0], result.data() + dst);
        }
    }

    for (int source = 1; source < static_cast<int>(volume(dims)); ++source) {
        Extent coords;
        MPI_Cart_coords(cart, source, 3, coords.data());
        const Domain remote = make_domain(work, dims, coords, hasInterior);
        Box target = remote.owned;
        for (int axis = 0; axis < 3; ++axis) {
            target.begin[axis] += remote.origin[axis];
            target.end[axis] += remote.origin[axis];
        }
        MPI_Datatype targetType = subarray(global, target);
        MPI_Recv(result.data(), 1, targetType, source, tag, cart, MPI_STATUS_IGNORE);
        MPI_Type_free(&targetType);
    }
    print_results(result, "Grid");
}

bool validate_result(const Real* grid, const Domain& domain, int rank, MPI_Comm cart) {
    // Negate the minimum so a single maximum reduction can combine all checks.
    const Real lowest = -std::numeric_limits<Real>::infinity();
    std::array<Real, 3> local{lowest, lowest, 0};
    const auto& [begin, end] = domain.owned;
    for (int z = begin[2]; z < end[2]; ++z) {
        for (int y = begin[1]; y < end[1]; ++y) {
            const Real* row = grid + index_of(0, y, z, domain.shape);
            for (int x = begin[0]; x < end[0]; ++x) {
                const Real value = row[x];
                local[0] = std::max(local[0], -value);
                local[1] = std::max(local[1], value);
                if (!std::isfinite(value)) local[2] = 1;
            }
        }
    }

    std::array<Real, 3> summary;
    MPI_Allreduce(local.data(), summary.data(), 3, MPI_DOUBLE, MPI_MAX, cart);
    if (summary[2] != 0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const bool valid = summary[0] <= 1e6 && summary[1] <= 1e6;
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", -summary[0], summary[1]);
        if (!valid) printf("Validation failed: values out of expected range\n");
    }
    return valid;
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

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    Extent global{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            global[0] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            global[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            global[2] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    if (*std::min_element(global.begin(), global.end()) <= 0) {
        if (rank == 0) printf("Grid dimensions must be positive\n");
        return 1;
    }
    if (static_cast<size_t>(global[0]) >
        std::numeric_limits<size_t>::max() / global[1] / global[2]) {
        if (rank == 0) printf("Grid dimensions are too large\n");
        return 1;
    }

    const bool hasInterior = *std::min_element(global.begin(), global.end()) > 2;
    const Extent work = hasInterior ?
        Extent{global[0] - 2, global[1] - 2, global[2] - 2} : global;
    Extent dims;
    if (rank == 0) dims = process_grid(work, ranks);
    MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    const Extent periods{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    Extent coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    const Domain domain = make_domain(work, dims, coords, hasInterior);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const size_t localSize = volume(domain.shape);
    auto first = std::make_unique_for_overwrite<Real[]>(localSize);
    auto second = std::make_unique_for_overwrite<Real[]>(localSize);
    const std::array<Real*, 2> grids{first.get(), second.get()};
    initialize_grids(grids[0], grids[1], domain, global, iterations, hasInterior);

    if (rank == 0) printf("Running stencil computation...\n");
    double elapsed;
    if (hasInterior && iterations > 0) {
        HaloExchange halo(domain, cart, grids);
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            const int buffer = iter % 2;
            const Real* input = grids[buffer];
            Real* output = grids[1 - buffer];
            auto& requests = halo.requests[buffer];

            if (halo.count > 0) MPI_Startall(halo.count, requests.data());
            stencil(input, output, domain.shape, halo.core);
            if (halo.count > 0) MPI_Waitall(halo.count, requests.data(), MPI_STATUSES_IGNORE);
            for (int face = 0; face < halo.faces; ++face) {
                stencil(input, output, domain.shape, halo.boundary[face]);
            }
        }
        elapsed = MPI_Wtime() - start;
    } else {
        elapsed = 0;
    }

    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000));
        const double updates = hasInterior ? static_cast<double>(volume(work)) *
                                            std::max(iterations, 0) : 0;
        printf("Performance: %.3f MCellUpdates/s\n",
               duration > 0 ? updates / duration / 1e6 : 0);
    }

    const Real* finalGrid = grids[iterations % 2 != 0];
    if (printResults) {
        print_global_results(finalGrid, domain, global, work, dims, hasInterior, rank, cart);
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validate_result(finalGrid, domain, rank, cart);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
