#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using Grid = std::array<int, 3>;

struct Box {
    Grid begin;
    Grid end;
};

struct Block {
    Grid start;
    Grid size;
};

Block partition(const Grid& global, const Grid& dims, const Grid& coords) {
    Block block{};
    for (int axis = 0; axis < 3; ++axis) {
        const int base = global[axis] / dims[axis];
        const int extra = global[axis] % dims[axis];
        block.start[axis] = coords[axis] * base + std::min(coords[axis], extra);
        block.size[axis] = base + (coords[axis] < extra);
    }
    return block;
}

Grid processGrid(const Grid& global, int ranks, size_t volume) {
    ranks = static_cast<int>(std::min(volume, static_cast<size_t>(ranks)));
    for (int count = ranks; count > 0; --count) {
        Grid best{};
        double bestCost = std::numeric_limits<double>::infinity();
        for (int px = 1; px <= std::min(global[0], count); ++px) {
            if (count % px != 0) continue;
            const int yz = count / px;
            for (int py = 1; py <= std::min(global[1], yz); ++py) {
                if (yz % py != 0) continue;
                const int pz = yz / py;
                if (pz > global[2]) continue;

                const Grid dims{px, py, pz};
                double largestBlock = 1.0;
                double haloRatio = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    largestBlock *= 1 + (global[axis] - 1) / dims[axis];
                    haloRatio += static_cast<double>(dims[axis] - 1) / global[axis];
                }
                const double cost = largestBlock * (1.0 + 2.0 * haloRatio);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = dims;
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

MPI_Datatype subarray(const Grid& extent, const Grid& size, const Grid& start) {
    MPI_Datatype type;
    // The first coordinate (X) is contiguous in both local and global storage.
    MPI_Type_create_subarray(3, extent.data(), size.data(), start.data(),
                             MPI_ORDER_FORTRAN, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

class Domain {
public:
    const Grid size;
    const Grid extent;
    const size_t row;
    const size_t plane;
    const size_t storage;
    const MPI_Comm comm;

    Domain(const Grid& size, MPI_Comm comm)
        : size(size), extent{size[0] + 2, size[1] + 2, size[2] + 2},
          row(extent[0]), plane(row * extent[1]), storage(plane * extent[2]), comm(comm),
          interior{{1, 1, 1}, {size[0] + 1, size[1] + 1, size[2] + 1}} {
        for (int axis = 2; axis >= 0; --axis) {
            MPI_Cart_shift(comm, axis, 1, &neighbors[2 * axis], &neighbors[2 * axis + 1]);
            if (neighbors[2 * axis] != MPI_PROC_NULL || neighbors[2 * axis + 1] != MPI_PROC_NULL) {
                Grid face = size;
                face[axis] = 1;
                faceTypes[axis] = subarray(extent, face, {0, 0, 0});
            }

            const size_t stride = std::array<size_t, 3>{1, row, plane}[axis];
            for (int side = 0; side < 2; ++side) {
                const int face = 2 * axis + side;
                Grid first{1, 1, 1};
                first[axis] = side == 0 ? 1 : size[axis];
                sendOffset[face] = index(first[0], first[1], first[2]);
                recvOffset[face] = side == 0 ? sendOffset[face] - stride : sendOffset[face] + stride;
                if (neighbors[face] == MPI_PROC_NULL || interior.begin[axis] == interior.end[axis]) continue;

                // Peel off disjoint boundary boxes, including blocks only one cell thick.
                Box edge = interior;
                if (side == 0) {
                    edge.end[axis] = ++interior.begin[axis];
                } else {
                    edge.begin[axis] = --interior.end[axis];
                }
                boundary[boundaryCount++] = edge;
            }
        }
    }

    ~Domain() {
        for (auto& type : faceTypes) {
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
        }
    }

    size_t index(int x, int y, int z) const {
        return z * plane + y * row + x;
    }

    struct HaloExchange {
        double* const field;
        std::array<MPI_Request, 12> requests;
        int count = 0;

        HaloExchange(const Domain& domain, double* field) : field(field) {
            for (int face = 0; face < 6; ++face) {
                if (domain.neighbors[face] != MPI_PROC_NULL) {
                    MPI_Recv_init(field + domain.recvOffset[face], 1, domain.faceTypes[face / 2],
                                  domain.neighbors[face], face ^ 1, domain.comm, &requests[count++]);
                }
            }
            for (int face = 0; face < 6; ++face) {
                if (domain.neighbors[face] != MPI_PROC_NULL) {
                    MPI_Send_init(field + domain.sendOffset[face], 1, domain.faceTypes[face / 2],
                                  domain.neighbors[face], face, domain.comm, &requests[count++]);
                }
            }
        }

        ~HaloExchange() {
            for (int i = 0; i < count; ++i) MPI_Request_free(&requests[i]);
        }

        HaloExchange(const HaloExchange&) = delete;
        HaloExchange& operator=(const HaloExchange&) = delete;
    };

    template <typename Update>
    void stencil(HaloExchange& halo, const Update& update) const {
        if (halo.count != 0) MPI_Startall(halo.count, halo.requests.data());
        for (int face = 0; face < 6; ++face) {
            if (neighbors[face] == MPI_PROC_NULL) clampFace(halo.field, face);
        }

        compute(halo.field, interior, update);
        if (halo.count != 0) MPI_Waitall(halo.count, halo.requests.data(), MPI_STATUSES_IGNORE);
        for (int i = 0; i < boundaryCount; ++i) {
            compute(halo.field, boundary[i], update);
        }
    }

private:
    std::array<int, 6> neighbors{};
    std::array<size_t, 6> sendOffset{};
    std::array<size_t, 6> recvOffset{};
    std::array<MPI_Datatype, 3> faceTypes{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    Box interior;
    std::array<Box, 6> boundary{};
    int boundaryCount = 0;

    void clampFace(double* field, int face) const {
        const double* source = field + sendOffset[face];
        double* ghost = field + recvOffset[face];
        switch (face / 2) {
        case 0:
            for (int z = 0; z < size[2]; ++z) {
                for (int y = 0; y < size[1]; ++y) {
                    ghost[z * plane + y * row] = source[z * plane + y * row];
                }
            }
            break;
        case 1:
            for (int z = 0; z < size[2]; ++z) {
                std::copy_n(source + z * plane, size[0], ghost + z * plane);
            }
            break;
        case 2:
            for (int y = 0; y < size[1]; ++y) {
                std::copy_n(source + y * row, size[0], ghost + y * row);
            }
            break;
        }
    }

    template <typename Update>
    void compute(const double* field, const Box& box, const Update& update) const {
        for (int z = box.begin[2]; z < box.end[2]; ++z) {
            for (int y = box.begin[1]; y < box.end[1]; ++y) {
                const size_t end = index(box.end[0], y, z);
                for (size_t i = index(box.begin[0], y, z); i < end; ++i) {
                    const double center = field[i];
                    const double xx = field[i + 1] + field[i - 1] - 2.0 * center;
                    const double yy = field[i + row] + field[i - row] - 2.0 * center;
                    const double zz = field[i + plane] + field[i - plane] - 2.0 * center;
                    update(i, center, xx + yy + zz);
                }
            }
        }
    }
};

void initializeConcentration(double* c, const Domain& domain, const Block& block,
                             const Grid& global, size_t volume) {
    const size_t globalPlane = static_cast<size_t>(global[0]) * global[1];
    for (int z = 0; z < block.size[2]; ++z) {
        for (int y = 0; y < block.size[1]; ++y) {
            const size_t first = (block.start[2] + z) * globalPlane
                               + (block.start[1] + y) * static_cast<size_t>(global[0])
                               + block.start[0];
            const size_t local = domain.index(1, y + 1, z + 1);
            for (int x = 0; x < block.size[0]; ++x) {
                const double pseudo = ((first + x + 1) * 1299709) % volume / static_cast<double>(volume);
                c[local + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void printDistributedResults(const double* c, const Domain& domain, const Grid& global,
                             const Grid& dims, size_t volume, int rank, int ranks) {
    // Receive directly into global order for the original ordered sum and hash.
    if (rank == 0) {
        std::vector<double> result(volume);
        std::vector<MPI_Request> requests(ranks - 1);
        for (int source = 1; source < ranks; ++source) {
            Grid coords;
            MPI_Cart_coords(domain.comm, source, 3, coords.data());
            const Block block = partition(global, dims, coords);
            MPI_Datatype destination = subarray(global, block.size, block.start);
            MPI_Irecv(result.data(), 1, destination, source, 0, domain.comm, &requests[source - 1]);
            MPI_Type_free(&destination);
        }
        const size_t globalPlane = static_cast<size_t>(global[0]) * global[1];
        for (int z = 0; z < domain.size[2]; ++z) {
            for (int y = 0; y < domain.size[1]; ++y) {
                std::copy_n(c + domain.index(1, y + 1, z + 1), domain.size[0],
                            result.data() + z * globalPlane + y * static_cast<size_t>(global[0]));
            }
        }
        MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
        print_results(result, "Concentration");
    } else {
        MPI_Datatype owned = subarray(domain.extent, domain.size, {1, 1, 1});
        MPI_Send(c, 1, owned, 0, 0, domain.comm);
        MPI_Type_free(&owned);
    }
}

bool validateResult(const double* c, const Domain& domain, int rank) {
    const double infinity = std::numeric_limits<double>::infinity();
    double local[3] = {-infinity, -infinity, 0.0};
    for (int z = 1; z <= domain.size[2]; ++z) {
        for (int y = 1; y <= domain.size[1]; ++y) {
            const double* values = c + domain.index(1, y, z);
            for (int x = 0; x < domain.size[0]; ++x) {
                const double value = values[x];
                local[0] = std::max(local[0], -value);
                local[1] = std::max(local[1], value);
                local[2] = std::max(local[2], static_cast<double>(!std::isfinite(value)));
            }
        }
    }
    double global[3];
    MPI_Reduce(local, global, 3, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (rank != 0) return true;
    if (global[2] != 0.0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const double minValue = -global[0];
    const double maxValue = global[1];
    printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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

int runBenchmark(const Grid& global, const Grid& dims, size_t volume, int iterations,
                 bool validate, bool printResults, MPI_Comm comm, int ranks) {
    int rank;
    Grid coords;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords.data());
    const Block block = partition(global, dims, coords);
    const Domain domain(block.size, comm);

    auto cold = std::make_unique_for_overwrite<double[]>(domain.storage);
    auto mu = std::make_unique<double[]>(domain.storage);
    initializeConcentration(cold.get(), domain, block, global, volume);
    Domain::HaloExchange concentrationHalo(domain, cold.get());
    Domain::HaloExchange potentialHalo(domain, mu.get());

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    constexpr double dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double D = 1.0;

    for (int t = 0; t < iterations; ++t) {
        domain.stencil(concentrationHalo, [&](size_t i, double cv, double laplacian) {
            mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv - gamma * laplacian;
        });
        // The concentration stencil has finished, so each update only needs
        // the previous concentration at the cell it is about to overwrite.
        domain.stencil(potentialHalo, [&](size_t i, double, double laplacian) {
            cold[i] = cold[i] + dt * D * laplacian;
        });
    }

    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double mcups = duration > 0.0 ? static_cast<double>(volume) * iterations / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) printDistributedResults(cold.get(), domain, global, dims, volume, rank, ranks);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold.get(), domain, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int benchmarkMain(int argc, char** argv) {
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    Grid global{64, 0, 0};
    int iterations = 20;
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

    size_t paddedVolume = 1;
    for (int n : global) {
        if (n <= 0 || n > std::numeric_limits<int>::max() - 2 ||
            paddedVolume > std::numeric_limits<size_t>::max() / sizeof(double) / (n + 2)) {
            if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
            return 1;
        }
        paddedVolume *= n + 2;
    }
    const size_t volume = static_cast<size_t>(global[0]) * global[1] * global[2];
    const Grid dims = processGrid(global, ranks, volume);
    const int activeRanks = dims[0] * dims[1] * dims[2];

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    MPI_Comm cart;
    const Grid periodic{0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periodic.data(), 0, &cart);
    int status = 0;
    if (cart != MPI_COMM_NULL) {
        status = runBenchmark(global, dims, volume, iterations, validate, printResults, cart, activeRanks);
        MPI_Comm_free(&cart);
    }
    if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = benchmarkMain(argc, argv);
    MPI_Finalize();
    return status;
}
