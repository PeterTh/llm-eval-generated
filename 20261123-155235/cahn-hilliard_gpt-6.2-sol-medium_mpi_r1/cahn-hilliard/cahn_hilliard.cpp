#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

using Shape = std::array<int, 3>;

struct Layout {
    size_t row;
    size_t plane;

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }
};

struct Block {
    Shape start;
    Shape extent;
};

Shape chooseProcessGrid(const Shape& grid, int ranks) {
    Shape best{1, 1, 1};
    int mostRanks = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    // Prefer using all ranks. Among equally sized process grids, limit the
    // largest block's work and halo area. Favor wider rows on a tie.
    for (int px = 1; px <= std::min(grid[0], ranks); ++px) {
        for (int py = 1; py <= std::min(grid[1], ranks / px); ++py) {
            const int pz = std::min(grid[2], ranks / (px * py));
            const int used = px * py * pz;
            if (used < mostRanks) continue;

            const double x = 1 + (grid[0] - 1) / px;
            const double y = 1 + (grid[1] - 1) / py;
            const double z = 1 + (grid[2] - 1) / pz;
            const double cost = x * y * z
                              + (px > 1 ? y * z : 0)
                              + (py > 1 ? x * z : 0)
                              + (pz > 1 ? x * y : 0);
            if (used > mostRanks || cost < bestCost) {
                best = {px, py, pz};
                mostRanks = used;
                bestCost = cost;
            }
        }
    }
    return best;
}

Block rankBlock(MPI_Comm comm, int rank, const Shape& grid, const Shape& processes) {
    Shape coords;
    MPI_Cart_coords(comm, rank, 3, coords.data());
    Block block;
    for (int axis = 0; axis < 3; ++axis) {
        const int base = grid[axis] / processes[axis];
        const int extra = grid[axis] % processes[axis];
        block.start[axis] = coords[axis] * base + std::min(coords[axis], extra);
        block.extent[axis] = base + (coords[axis] < extra);
    }
    return block;
}

// Describe the same rectangular region in a local padded or global dense array.
MPI_Datatype blockType(const Layout& layout, const Shape& extent) {
    MPI_Datatype rows, block;
    MPI_Type_create_hvector(extent[1], extent[0], layout.row * sizeof(double), MPI_DOUBLE, &rows);
    MPI_Type_create_hvector(extent[2], 1, layout.plane * sizeof(double), rows, &block);
    MPI_Type_commit(&block);
    MPI_Type_free(&rows);
    return block;
}

using Cell = std::array<size_t, 3>;

struct Region {
    Cell begin, end;

    bool empty() const {
        return begin[0] == end[0] || begin[1] == end[1] || begin[2] == end[2];
    }

    template <typename Function>
    void forRows(const Layout& layout, Function function) const {
        const size_t width = end[0] - begin[0];
        for (size_t z = begin[2]; z < end[2]; ++z) {
            for (size_t y = begin[1]; y < end[1]; ++y) {
                function(layout.index(begin[0], y, z), width);
            }
        }
    }
};

struct Face {
    int neighbor;
    Region owned;
    size_t send, receive;
};

struct StencilDomain {
    Layout layout;
    Region owned, interior;
    std::array<Region, 6> boundary;
    std::array<Face, 6> faces;
    int boundaryCount = 0;

    StencilDomain(MPI_Comm comm, const Shape& extent)
        : layout{size_t(extent[0]) + 2, (size_t(extent[0]) + 2) * (size_t(extent[1]) + 2)},
          owned{{1, 1, 1}, {size_t(extent[0]) + 1, size_t(extent[1]) + 1, size_t(extent[2]) + 1}},
          interior(owned) {
        // Peel only faces that need MPI data, assigning edge and corner cells
        // to the first applicable region so each stencil output is computed once.
        for (int axis = 2; axis >= 0; --axis) {
            int neighbors[2];
            MPI_Cart_shift(comm, axis, 1, &neighbors[0], &neighbors[1]);
            for (int side = 0; side < 2; ++side) {
                const size_t edge = side == 0 ? 1 : extent[axis];
                Cell send = owned.begin;
                send[axis] = edge;
                Cell receive = send;
                receive[axis] = side == 0 ? 0 : edge + 1;
                Region face = owned;
                face.begin[axis] = edge;
                face.end[axis] = edge + 1;
                faces[2 * axis + side] = {neighbors[side], face,
                    layout.index(send[0], send[1], send[2]),
                    layout.index(receive[0], receive[1], receive[2])};

                if (neighbors[side] == MPI_PROC_NULL || interior.empty()) continue;
                Region shell = interior;
                shell.begin[axis] = edge;
                shell.end[axis] = edge + 1;
                boundary[boundaryCount++] = shell;
                if (side == 0) ++interior.begin[axis];
                else --interior.end[axis];
            }
        }
    }

    void clampPhysicalFaces(double* field) const {
        for (int f = 0; f < 6; ++f) {
            const Face& face = faces[f];
            if (face.neighbor != MPI_PROC_NULL) continue;
            double* ghost = field + face.receive;
            if (f < 2) {
                face.owned.forRows(layout, [&](size_t offset, size_t) {
                    ghost[offset - face.send] = field[offset];
                });
            } else {
                face.owned.forRows(layout, [&](size_t offset, size_t width) {
                    std::copy_n(field + offset, width, ghost + (offset - face.send));
                });
            }
        }
    }
};

class HaloExchange {
public:
    void bind(MPI_Comm comm, const StencilDomain& domain,
              const std::array<MPI_Datatype, 3>& types, double* field) {
        // Receive into ghost faces and send from owned faces.
        for (int f = 0; f < 6; ++f) {
            const Face& face = domain.faces[f];
            if (face.neighbor == MPI_PROC_NULL) continue;
            MPI_Recv_init(field + face.receive, 1, types[f / 2], face.neighbor, 0, comm, &requests[count++]);
        }
        for (int f = 0; f < 6; ++f) {
            const Face& face = domain.faces[f];
            if (face.neighbor == MPI_PROC_NULL) continue;
            MPI_Send_init(field + face.send, 1, types[f / 2], face.neighbor, 0, comm, &requests[count++]);
        }
    }

    void start() {
        if (count != 0) MPI_Startall(count, requests.data());
    }

    void wait() {
        if (count != 0) MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
    }

    ~HaloExchange() {
        for (int i = 0; i < count; ++i) MPI_Request_free(&requests[i]);
    }

private:
    std::array<MPI_Request, 12> requests;
    int count = 0;
};

template <typename Function>
void applyStencil(const StencilDomain& domain, HaloExchange& exchange, double* field,
                  Function computeRow) {
    exchange.start();
    domain.clampPhysicalFaces(field);
    if (!domain.interior.empty()) domain.interior.forRows(domain.layout, computeRow);
    exchange.wait();
    for (int i = 0; i < domain.boundaryCount; ++i) {
        domain.boundary[i].forRows(domain.layout, computeRow);
    }
}

// The benchmark uses unit spacing in all three dimensions.
inline double laplacian(const double* center, size_t x, const Layout& layout) {
    const double twiceCenter = 2.0 * center[x];
    const double cxx = center[x + 1] + (center - 1)[x] - twiceCenter;
    const double cyy = (center + layout.row)[x] + (center - layout.row)[x] - twiceCenter;
    const double czz = (center + layout.plane)[x] + (center - layout.plane)[x] - twiceCenter;
    return cxx + cyy + czz;
}

void gatherResults(MPI_Comm comm, int rank, const Shape& grid, const Shape& processes,
                   const Block& local, const StencilDomain& domain, const double* field,
                   size_t gridSize) {
    const Layout global{size_t(grid[0]), size_t(grid[0]) * grid[1]};
    std::vector<double> result;
    std::vector<MPI_Request> receives;
    constexpr int resultTag = 1;
    if (rank == 0) {
        result.resize(gridSize);
        int ranks;
        MPI_Comm_size(comm, &ranks);
        receives.resize(ranks);
        for (int source = 0; source < ranks; ++source) {
            const Block block = rankBlock(comm, source, grid, processes);
            MPI_Datatype type = blockType(global, block.extent);
            const size_t offset = global.index(block.start[0], block.start[1], block.start[2]);
            MPI_Irecv(result.data() + offset, 1, type, source, resultTag, comm, &receives[source]);
            MPI_Type_free(&type);
        }
    }

    MPI_Datatype type = blockType(domain.layout, local.extent);
    MPI_Send(field + domain.layout.index(1, 1, 1), 1, type, 0, resultTag, comm);
    MPI_Type_free(&type);

    if (rank == 0) {
        MPI_Waitall(receives.size(), receives.data(), MPI_STATUSES_IGNORE);
        print_results(result, "Concentration");
    }
}

bool validateResult(MPI_Comm comm, int rank, const StencilDomain& domain, const double* field) {
    // One minimum reduction yields the range and detects nonfinite values.
    double local[3] = {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(), 0.0};
    domain.owned.forRows(domain.layout, [&](size_t offset, size_t width) {
        for (size_t x = 0; x < width; ++x) {
            const double value = field[offset + x];
            if (std::isfinite(value)) {
                local[0] = std::min(local[0], value);
                local[1] = std::min(local[1], -value);
            } else {
                local[2] = -1.0;
            }
        }
    });
    double reduced[3];
    MPI_Allreduce(local, reduced, 3, MPI_DOUBLE, MPI_MIN, comm);
    const double minimum = reduced[0];
    const double maximum = -reduced[1];
    const bool finite = reduced[2] == 0.0;
    const bool valid = finite && minimum >= -10.0 && maximum <= 10.0;
    if (rank == 0) {
        if (!finite) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
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

int simulate(MPI_Comm comm, int rank, const Shape& grid, const Shape& processes,
             int iterations, bool validate, bool printResults) {
    const size_t gridSize = size_t(grid[0]) * grid[1] * grid[2];
    const Block block = rankBlock(comm, rank, grid, processes);
    const StencilDomain domain(comm, block.extent);
    const Layout& layout = domain.layout;
    const size_t storage = layout.plane * (size_t(block.extent[2]) + 2);
    std::vector<double> concentration[2] = {std::vector<double>(storage), std::vector<double>(storage)};
    std::vector<double> mu(storage);

    if (rank == 0) printf("Initializing concentration field...\n");
    for (int z = 0; z < block.extent[2]; ++z) {
        for (int y = 0; y < block.extent[1]; ++y) {
            const size_t globalRow = (size_t(block.start[2] + z) * grid[1] + block.start[1] + y)
                                   * grid[0] + block.start[0];
            double* row = concentration[0].data() + layout.index(1, y + 1, z + 1);
            for (int x = 0; x < block.extent[0]; ++x) {
                const double pseudo = ((globalRow + x + 1) * 1299709 % gridSize) / double(gridSize);
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    std::array<MPI_Datatype, 3> types;
    for (int axis = 0; axis < 3; ++axis) {
        Shape face = block.extent;
        face[axis] = 1;
        types[axis] = MPI_DATATYPE_NULL;
        if (processes[axis] > 1) types[axis] = blockType(layout, face);
    }
    HaloExchange concentrationHalo[2], muHalo;
    concentrationHalo[0].bind(comm, domain, types, concentration[0].data());
    concentrationHalo[1].bind(comm, domain, types, concentration[1].data());
    muHalo.bind(comm, domain, types, mu.data());
    for (auto& type : types) {
        if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
    }

    constexpr double dt = 0.01, diffusion = 1.0, gamma = 0.5;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    int current = 0;
    double* potential = mu.data();
    for (int t = 0; t < iterations; ++t) {
        double* cold = concentration[current].data();
        double* cnew = concentration[1 - current].data();
        applyStencil(domain, concentrationHalo[current], cold, [&](size_t offset, size_t width) {
            const double* center = cold + offset;
            double* output = potential + offset;
            for (size_t x = 0; x < width; ++x) {
                const double cv = center[x];
                output[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv - gamma * laplacian(center, x, layout);
            }
        });
        applyStencil(domain, muHalo, potential, [&](size_t offset, size_t width) {
            const double* center = potential + offset;
            const double* previous = cold + offset;
            double* output = cnew + offset;
            for (size_t x = 0; x < width; ++x) {
                output[x] = previous[x] + dt * diffusion * laplacian(center, x, layout);
            }
        });
        current = 1 - current;
    }

    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", double(gridSize) * iterations / elapsed / 1e6);
    }
    const double* result = concentration[current].data();
    if (printResults) gatherResults(comm, rank, grid, processes, block, domain, result, gridSize);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        return validateResult(comm, rank, domain, result) ? 0 : 1;
    }
    return 0;
}

int run(int argc, char** argv, int rank, int ranks) {
    Shape grid{64, 0, 0};
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            grid[0] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            grid[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            grid[2] = atoi(argv[++i]);
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

    if (grid[1] == 0) grid[1] = grid[0];
    if (grid[2] == 0) grid[2] = grid[0];
    if (std::any_of(grid.begin(), grid.end(), [](int length) { return length < 0; })) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be nonnegative\n");
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", grid[0], grid[1], grid[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    if (grid[0] == 0) {
        if (rank == 0) {
            printf("Initializing concentration field...\n");
            printf("Running Cahn-Hilliard simulation...\n");
            printf("Computation time: 0 ms\n");
            printf("Performance: 0.000 MCellUpdates/s\n");
            if (printResults) print_results(std::vector<double>{}, "Concentration");
            if (validate) {
                printf("Validating result...\n");
                printf("Validation failed: empty concentration field\n");
                printf("Validation: FAILED\n");
            }
        }
        return validate ? 1 : 0;
    }

    const Shape processes = chooseProcessGrid(grid, ranks);
    const Shape periods{0, 0, 0};
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, processes.data(), periods.data(), 0, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        status = simulate(comm, rank, grid, processes, iterations, validate, printResults);
        MPI_Comm_free(&comm);
    }
    return status;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
