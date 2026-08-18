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

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 1;
    std::array<int, 3> dims{}, coords{}; // x, y, z
    std::array<int, 6> neighbor{};       // x-, x+, y-, y+, z-, z+
    std::array<size_t, 3> global{}, first{}, local{};
    size_t sx = 0, sy = 0;
    std::array<MPI_Datatype, 6> send_type{}, recv_type{};

    size_t index(size_t x, size_t y, size_t z) const noexcept {
        return (z * sy + y) * sx + x;
    }
};

static void split_dimension(size_t n, int parts, int coordinate,
                            size_t& first, size_t& count) {
    const size_t q = n / static_cast<size_t>(parts);
    const size_t r = n % static_cast<size_t>(parts);
    count = q + (static_cast<size_t>(coordinate) < r);
    first = static_cast<size_t>(coordinate) * q +
            std::min(static_cast<size_t>(coordinate), r);
}

// Choose a grid that minimizes the total area of internal process interfaces.
static std::array<int, 3> process_grid(int nranks, const std::array<size_t, 3>& n) {
    std::array<int, 3> best{0, 0, 0};
    long double best_cost = std::numeric_limits<long double>::infinity();
    long double best_imbalance = std::numeric_limits<long double>::infinity();
    for (int px = 1; px <= nranks; ++px) {
        if (nranks % px != 0 || static_cast<size_t>(px) > n[0]) continue;
        const int left = nranks / px;
        for (int py = 1; py <= left; ++py) {
            if (left % py != 0 || static_cast<size_t>(py) > n[1]) continue;
            const int pz = left / py;
            if (static_cast<size_t>(pz) > n[2]) continue;
            const long double cost = static_cast<long double>(px - 1) * n[1] * n[2] +
                                     static_cast<long double>(py - 1) * n[0] * n[2] +
                                     static_cast<long double>(pz - 1) * n[0] * n[1];
            const long double lx = static_cast<long double>(n[0]) / px;
            const long double ly = static_cast<long double>(n[1]) / py;
            const long double lz = static_cast<long double>(n[2]) / pz;
            const long double imbalance = std::max({lx, ly, lz}) / std::min({lx, ly, lz});
            if (cost < best_cost || (cost == best_cost && imbalance < best_imbalance)) {
                best = {px, py, pz};
                best_cost = cost;
                best_imbalance = imbalance;
            }
        }
    }
    return best;
}

static MPI_Datatype subarray_type(const Domain& d, int axis, int side, bool receive) {
    int sizes[3] = {static_cast<int>(d.local[2] + 2),
                    static_cast<int>(d.local[1] + 2),
                    static_cast<int>(d.local[0] + 2)};
    int subs[3] = {static_cast<int>(d.local[2]),
                   static_cast<int>(d.local[1]),
                   static_cast<int>(d.local[0])};
    int starts[3] = {1, 1, 1};
    const int a = 2 - axis; // MPI subarray is ordered z, y, x
    subs[a] = 1;
    if (receive)
        starts[a] = side == 0 ? 0 : static_cast<int>(d.local[axis] + 1);
    else
        starts[a] = side == 0 ? 1 : static_cast<int>(d.local[axis]);
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

static bool create_domain(Domain& d, const std::array<size_t, 3>& global) {
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    d.global = global;
    d.dims = process_grid(d.size, global);
    if (d.dims[0] == 0) return false;
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims.data(), periods, 0, &d.comm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Cart_coords(d.comm, d.rank, 3, d.coords.data());
    for (int axis = 0; axis < 3; ++axis) {
        split_dimension(global[axis], d.dims[axis], d.coords[axis],
                        d.first[axis], d.local[axis]);
        MPI_Cart_shift(d.comm, axis, 1, &d.neighbor[2 * axis], &d.neighbor[2 * axis + 1]);
    }
    d.sx = d.local[0] + 2;
    d.sy = d.local[1] + 2;
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side) {
            const int face = 2 * axis + side;
            d.send_type[face] = subarray_type(d, axis, side, false);
            d.recv_type[face] = subarray_type(d, axis, side, true);
        }
    return true;
}

static void destroy_domain(Domain& d) {
    for (MPI_Datatype& t : d.send_type) MPI_Type_free(&t);
    for (MPI_Datatype& t : d.recv_type) MPI_Type_free(&t);
    MPI_Comm_free(&d.comm);
}

static void fill_physical_halos(std::vector<double>& a, const Domain& d) {
    if (d.neighbor[0] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.local[2]; ++z) for (size_t y = 1; y <= d.local[1]; ++y)
            a[d.index(0, y, z)] = a[d.index(1, y, z)];
    if (d.neighbor[1] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.local[2]; ++z) for (size_t y = 1; y <= d.local[1]; ++y)
            a[d.index(d.local[0] + 1, y, z)] = a[d.index(d.local[0], y, z)];
    if (d.neighbor[2] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.local[2]; ++z) for (size_t x = 1; x <= d.local[0]; ++x)
            a[d.index(x, 0, z)] = a[d.index(x, 1, z)];
    if (d.neighbor[3] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.local[2]; ++z) for (size_t x = 1; x <= d.local[0]; ++x)
            a[d.index(x, d.local[1] + 1, z)] = a[d.index(x, d.local[1], z)];
    if (d.neighbor[4] == MPI_PROC_NULL)
        for (size_t y = 1; y <= d.local[1]; ++y) for (size_t x = 1; x <= d.local[0]; ++x)
            a[d.index(x, y, 0)] = a[d.index(x, y, 1)];
    if (d.neighbor[5] == MPI_PROC_NULL)
        for (size_t y = 1; y <= d.local[1]; ++y) for (size_t x = 1; x <= d.local[0]; ++x)
            a[d.index(x, y, d.local[2] + 1)] = a[d.index(x, y, d.local[2])];
}

static void begin_exchange(std::vector<double>& a, const Domain& d,
                           std::array<MPI_Request, 12>& req) {
    fill_physical_halos(a, d);
    int q = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const int lo = 2 * axis, hi = lo + 1;
        MPI_Irecv(a.data(), 1, d.recv_type[lo], d.neighbor[lo], hi, d.comm, &req[q++]);
        MPI_Irecv(a.data(), 1, d.recv_type[hi], d.neighbor[hi], lo, d.comm, &req[q++]);
        MPI_Isend(a.data(), 1, d.send_type[lo], d.neighbor[lo], lo, d.comm, &req[q++]);
        MPI_Isend(a.data(), 1, d.send_type[hi], d.neighbor[hi], hi, d.comm, &req[q++]);
    }
}

static inline double laplacian(const double* a, size_t p, const Domain& d) noexcept {
    return (a[p + 1] + a[p - 1] - 2.0 * a[p]) +
           (a[p + d.sx] + a[p - d.sx] - 2.0 * a[p]) +
           (a[p + d.sx * d.sy] + a[p - d.sx * d.sy] - 2.0 * a[p]);
}

template <class Operation>
static void compute_interior_and_boundary(const Domain& d, Operation op,
                                          std::array<MPI_Request, 12>& req) {
    // Work independent of halos overlaps network progress.
    for (size_t z = 2; z < d.local[2]; ++z)
        for (size_t y = 2; y < d.local[1]; ++y)
            for (size_t x = 2; x < d.local[0]; ++x) op(x, y, z);
    MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
    for (size_t z = 1; z <= d.local[2]; ++z)
        for (size_t y = 1; y <= d.local[1]; ++y)
            for (size_t x = 1; x <= d.local[0]; ++x)
                if (x == 1 || x == d.local[0] || y == 1 || y == d.local[1] ||
                    z == 1 || z == d.local[2]) op(x, y, z);
}

static void initialize(std::vector<double>& c, const Domain& d) {
    const size_t volume = d.global[0] * d.global[1] * d.global[2];
    for (size_t z = 1; z <= d.local[2]; ++z)
        for (size_t y = 1; y <= d.local[1]; ++y)
            for (size_t x = 1; x <= d.local[0]; ++x) {
                const size_t gx = d.first[0] + x - 1, gy = d.first[1] + y - 1;
                const size_t gz = d.first[2] + z - 1;
                const size_t gid = (gz * d.global[1] + gy) * d.global[0] + gx;
                const double pseudo = (((gid + 1) * size_t{1299709}) % volume) /
                                      static_cast<double>(volume);
                c[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
}

static std::vector<double> gather_global(const std::vector<double>& c, const Domain& d) {
    const size_t count_sz = d.local[0] * d.local[1] * d.local[2];
    std::vector<double> packed(count_sz);
    size_t p = 0;
    for (size_t z = 1; z <= d.local[2]; ++z)
        for (size_t y = 1; y <= d.local[1]; ++y)
            for (size_t x = 1; x <= d.local[0]; ++x) packed[p++] = c[d.index(x, y, z)];
    int count = static_cast<int>(count_sz);
    std::vector<int> counts(d.rank == 0 ? d.size : 0), displs(d.rank == 0 ? d.size : 0);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, d.comm);
    std::vector<double> received;
    if (d.rank == 0) {
        int total = 0;
        for (int r = 0; r < d.size; ++r) { displs[r] = total; total += counts[r]; }
        received.resize(static_cast<size_t>(total));
    }
    MPI_Gatherv(packed.data(), count, MPI_DOUBLE, received.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, d.comm);
    if (d.rank != 0) return {};
    std::vector<double> global(d.global[0] * d.global[1] * d.global[2]);
    for (int r = 0; r < d.size; ++r) {
        int coord[3]; MPI_Cart_coords(d.comm, r, 3, coord);
        size_t first[3], local[3];
        for (int a = 0; a < 3; ++a) split_dimension(d.global[a], d.dims[a], coord[a], first[a], local[a]);
        size_t q = static_cast<size_t>(displs[r]);
        for (size_t z = 0; z < local[2]; ++z)
            for (size_t y = 0; y < local[1]; ++y) {
                const size_t out = ((first[2] + z) * d.global[1] + first[1] + y) * d.global[0] + first[0];
                std::copy_n(received.data() + q, local[0], global.data() + out);
                q += local[0];
            }
    }
    return global;
}

static void print_usage(const char* name) {
    printf("Usage: %s [options]\n  -x <num> Grid X (default 64)\n  -y <num> Grid Y\n"
           "  -z <num> Grid Z\n  -i <num> Time steps (default 20)\n"
           "  -v       Validate\n  -r       Print results\n  -h       Help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_requested = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad) {
        if (world_rank == 0) { if (bad) printf("Invalid command line\n"); print_usage(argv[0]); }
        MPI_Finalize(); return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max() - 2) ||
        ny > static_cast<size_t>(std::numeric_limits<int>::max() - 2) ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) bad = true;
    Domain d;
    if (!bad && !create_domain(d, {nx, ny, nz})) bad = true;
    if (bad) {
        if (world_rank == 0) printf("Invalid grid/iteration count or more MPI ranks than grid cells\n");
        MPI_Finalize(); return 1;
    }
    if (d.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\nMPI ranks: %d (%d x %d x %d)\n",
               iterations, validate ? "enabled" : "disabled", d.size,
               d.dims[0], d.dims[1], d.dims[2]);
        printf("Initializing concentration field...\n");
    }
    const size_t allocation = (d.local[0] + 2) * (d.local[1] + 2) * (d.local[2] + 2);
    std::vector<double> cold(allocation), cnew(allocation), mu(allocation);
    initialize(cold, d);
    if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    const double start = MPI_Wtime();
    std::array<MPI_Request, 12> requests;
    for (int t = 0; t < iterations; ++t) {
        begin_exchange(cold, d, requests);
        compute_interior_and_boundary(d, [&](size_t x, size_t y, size_t z) {
            const size_t p = d.index(x, y, z);
            const double value = cold[p];
            mu[p] = 4.5 * ((value + 1.0) * (-(2.0 / 9.0)) +
                           (value - 1.0) * (-(2.0 / 9.0)) -
                           2.0 * value * (2.0 / 9.0)) +
                    3.0 * value + value * value * value - 0.5 * laplacian(cold.data(), p, d);
        }, requests);
        begin_exchange(mu, d, requests);
        compute_interior_and_boundary(d, [&](size_t x, size_t y, size_t z) {
            const size_t p = d.index(x, y, z);
            cnew[p] = cold[p] + 0.01 * laplacian(mu.data(), p, d);
        }, requests);
        cold.swap(cnew);
    }
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    if (d.rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double updates = static_cast<double>(nx) * ny * nz * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
    }
    if (print_results_requested) {
        std::vector<double> global = gather_global(cold, d);
        if (d.rank == 0) print_results(global, "Concentration");
    }
    int valid = 1;
    if (validate) {
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t z = 1; z <= d.local[2]; ++z)
            for (size_t y = 1; y <= d.local[1]; ++y)
                for (size_t x = 1; x <= d.local[0]; ++x) {
                    const double v = cold[d.index(x, y, z)];
                    if (!std::isfinite(v)) valid = 0;
                    local_min = std::min(local_min, v); local_max = std::max(local_max, v);
                }
        double global_min, global_max;
        int all_valid;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, d.comm);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
        MPI_Reduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, 0, d.comm);
        if (d.rank == 0) {
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", global_min, global_max);
            valid = all_valid && global_max <= 10.0 && global_min >= -10.0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, d.comm);
    }
    destroy_domain(d);
    MPI_Finalize();
    return valid ? 0 : 1;
}
