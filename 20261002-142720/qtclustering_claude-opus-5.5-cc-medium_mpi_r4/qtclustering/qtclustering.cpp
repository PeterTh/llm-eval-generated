// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Spatial grid used to accelerate neighbor queries. Cells are at least
// `threshold` wide, so every point within distance < threshold of a point
// lies in the 3x3 block of cells around it.
struct Grid {
    int nx = 1, ny = 1;
    double cell = 1.0;
    std::vector<int> start;   // CSR offsets, size nx*ny+1
    std::vector<int> items;   // point indices, ascending within each cell
    std::vector<int> cell_of; // cell id of each point

    // Only points with index % stride == offset are inserted
    void build(const std::vector<Point>& points, const double threshold,
               const int stride = 1, const int offset = 0) {
        const int N = static_cast<int>(points.size());
        const int MAX_CELLS = 2048;
        cell = std::max(threshold * 1.0001, std::max(MAX_WIDTH, MAX_HEIGHT) / MAX_CELLS);
        nx = std::max(1, std::min(MAX_CELLS, static_cast<int>(std::ceil(MAX_WIDTH / cell)) + 1));
        ny = std::max(1, std::min(MAX_CELLS, static_cast<int>(std::ceil(MAX_HEIGHT / cell)) + 1));
        start.assign(static_cast<size_t>(nx) * ny + 1, 0);
        cell_of.resize(N);
        for (int i = 0; i < N; ++i) {
            int cx = static_cast<int>(points[i].x / cell);
            int cy = static_cast<int>(points[i].y / cell);
            cx = std::min(std::max(cx, 0), nx - 1);
            cy = std::min(std::max(cy, 0), ny - 1);
            cell_of[i] = cy * nx + cx;
            if (i % stride == offset) start[cell_of[i] + 1]++;
        }
        for (size_t c = 0; c + 1 < start.size(); ++c) start[c + 1] += start[c];
        items.resize(start.back());
        std::vector<int> pos(start.begin(), start.end() - 1);
        for (int i = offset; i < N; i += stride) items[pos[cell_of[i]]++] = i;
    }

    template <class F>
    void forNeighborCells(const int p, F&& f) const {
        const int cx = cell_of[p] % nx;
        const int cy = cell_of[p] / nx;
        for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, ny - 1); ++y) {
            for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, nx - 1); ++x) {
                const int c = y * nx + x;
                for (int k = start[c]; k < start[c + 1]; ++k) f(items[k]);
            }
        }
    }
};

// Scratch buffers for candidate cluster generation (structure of arrays)
struct Workspace {
    std::vector<int> idx;
    std::vector<double> cx, cy, md2;
};

static const double DEAD = std::numeric_limits<double>::infinity();

// Smallest x >= 0 such that sqrt(x) >= t, so that sqrt(v) < t <=> v < x.
// Lets all distance comparisons run on squared distances while giving
// exactly the same decisions as comparing the (rounded) square roots.
static double sqrtLowerCut(const double t) {
    double x = t * t;
    while (x > 0.0 && std::sqrt(std::nextafter(x, 0.0)) >= t) x = std::nextafter(x, 0.0);
    while (std::sqrt(x) < t) x = std::nextafter(x, DEAD);
    return x;
}

// Smallest x > m such that sqrt(x) > sqrt(m): values in [m, x) have the
// same square root as m.
static double sqrtUpperCut(const double m) {
    const double s = std::sqrt(m);
    double x = std::nextafter(m, DEAD);
    while (x < DEAD && std::sqrt(x) == s) x = std::nextafter(x, DEAD);
    return x;
}

static inline double sqDist(const double ax, const double ay, const double bx, const double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

// 4-wide double vectors (GCC/Clang vector extension)
typedef double v4d __attribute__((vector_size(32)));
typedef long long v4l __attribute__((vector_size(32)));
static const int VW = 4;

static inline v4d vload(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline void vstore(double* p, const v4d v) { std::memcpy(p, &v, sizeof(v)); }
static inline v4d vsplat(const double x) { return v4d{x, x, x, x}; }

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster.
//
// Equivalent to the original greedy procedure: at each step the unclustered
// point with the smallest maximum distance to the current members (ties ->
// lowest index) is added as long as that distance is < threshold. The
// maximum (squared) distance of every candidate is maintained
// incrementally, and candidates whose maximum distance reaches the
// threshold are dropped permanently (it can only grow).
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const Grid& grid,
                             const double thr2, // sqrtLowerCut(threshold)
                             Workspace& ws,
                             std::vector<int>* cluster_members = nullptr) {
    const double sx = points[seed_point].x, sy = points[seed_point].y;
    ws.idx.clear();
    grid.forNeighborCells(seed_point, [&](int q) {
        if (q == seed_point || clustered[q]) return;
        if (sqDist(points[q].x, points[q].y, sx, sy) < thr2) ws.idx.push_back(q);
    });
    std::sort(ws.idx.begin(), ws.idx.end());

    int n = static_cast<int>(ws.idx.size());
    const int cap = (n + VW - 1) / VW * VW;
    ws.idx.resize(cap, -1);
    ws.cx.resize(cap);
    ws.cy.resize(cap);
    ws.md2.resize(cap);
    int* __restrict idx = ws.idx.data();
    double* __restrict px = ws.cx.data();
    double* __restrict py = ws.cy.data();
    double* __restrict md = ws.md2.data();
    double mn = DEAD;
    for (int i = 0; i < n; ++i) {
        px[i] = points[idx[i]].x;
        py[i] = points[idx[i]].y;
        md[i] = sqDist(px[i], py[i], sx, sy);
        mn = md[i] < mn ? md[i] : mn;
    }
    // Padding entries are permanently dead
    for (int i = n; i < cap; ++i) { px[i] = 0.0; py[i] = 0.0; md[i] = DEAD; }
    n = cap;

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }
    int size = 1;

    const v4d vthr = vsplat(thr2);
    const v4d vdead = vsplat(DEAD);
    while (mn < thr2) {
        // Select candidate with minimal max distance (first index on ties of
        // the rounded distance, i.e. squared values in [mn, upper))
        const double upper = sqrtUpperCut(mn);
        int best = 0;
        for (;; best += VW) {
            const v4l lt = vload(md + best) < vsplat(upper);
            if (lt[0] | lt[1] | lt[2] | lt[3]) {
                while (!(md[best] < upper)) ++best;
                break;
            }
        }
        const double bx = px[best], by = py[best];
        if (cluster_members) cluster_members->push_back(idx[best]);
        md[best] = DEAD;
        ++size;

        // Update max distances w.r.t. the new member; drop exceeded ones
        const v4d vbx = vsplat(bx), vby = vsplat(by);
        v4d vmin = vdead;
        v4l valive = v4l{0, 0, 0, 0};
        for (int i = 0; i < n; i += VW) {
            const v4d dx = vload(px + i) - vbx;
            const v4d dy = vload(py + i) - vby;
            const v4d d2 = dx * dx + dy * dy;
            v4d v = vload(md + i);
            v = v < d2 ? d2 : v;
            const v4l keep = v < vthr;
            v = keep ? v : vdead;
            vstore(md + i, v);
            vmin = v < vmin ? v : vmin;
            valive -= keep; // mask lanes are -1 when true
        }
        mn = std::min(std::min(vmin[0], vmin[1]), std::min(vmin[2], vmin[3]));
        const int alive = static_cast<int>(valive[0] + valive[1] + valive[2] + valive[3]);

        // Compact (order preserving) once many entries are dead
        if (alive * 2 < n) {
            int m = 0;
            for (int j = 0; j < n; ++j) {
                if (md[j] < DEAD) {
                    idx[m] = idx[j]; px[m] = px[j]; py[m] = py[j]; md[m] = md[j];
                    ++m;
                }
            }
            const int mcap = (m + VW - 1) / VW * VW;
            for (int j = m; j < mcap; ++j) { px[j] = 0.0; py[j] = 0.0; md[j] = DEAD; }
            n = mcap;
        }
    }
    return size;
}

// Main QT clustering algorithm (MPI-parallel: seeds are distributed
// cyclically over ranks; every rank ends up with the full result).
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    Grid grid, my_grid; // all points / seeds owned by this rank
    grid.build(points, threshold);
    my_grid.build(points, threshold, nprocs, rank);
    Workspace ws;
    const double thr2 = sqrtLowerCut(threshold);

    // Seeds owned by this rank (cyclic distribution). For each seed we keep
    // either its exact candidate cardinality (exact) or an upper bound on it
    // (initially 1 + #unclustered neighbors within threshold). Inexact seeds
    // are evaluated lazily, only when their bound could beat the current best.
    const int L = (N > rank) ? (N - rank + nprocs - 1) / nprocs : 0;
    std::vector<int> card(L, 0); // exact cardinality or upper bound
    std::vector<int> nb(L, 0);   // #unclustered neighbors within threshold
    std::vector<char> dirty(L, 1);
    std::vector<int> alive(L);
    for (int li = 0; li < L; ++li) {
        alive[li] = li;
        const int s = rank + li * nprocs;
        int cnt = 0;
        grid.forNeighborCells(s, [&](int q) {
            if (q != s && distance(points[q], points[s]) < threshold) ++cnt;
        });
        nb[li] = cnt;
        card[li] = cnt + 1;
    }

    struct ValLoc { int val; int loc; };
    // true if (v, s) beats b: larger cardinality, or equal with smaller seed
    auto beats = [](int v, int s, const ValLoc& b) {
        return v > b.val || (v == b.val && s < b.loc);
    };

    std::vector<int> pending;
    std::vector<int> members;
    int remaining = N;
    while (remaining > 0) {
        // Phase 1: best among clean (exactly known) seeds
        ValLoc local{-1, std::numeric_limits<int>::max()}, global;
        for (int li : alive) {
            if (!dirty[li] && beats(card[li], rank + li * nprocs, local)) {
                local = {card[li], rank + li * nprocs};
            }
        }
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Phase 2: evaluate dirty seeds whose bound could beat the best
        pending.clear();
        for (int li : alive) {
            if (dirty[li] && beats(card[li], rank + li * nprocs, global)) pending.push_back(li);
        }
        std::sort(pending.begin(), pending.end(), [&](int x, int y) {
            return card[x] != card[y] ? card[x] > card[y] : x < y;
        });
        ValLoc cur = beats(local.val, local.loc, global) ? local : global;
        for (int li : pending) {
            const int s = rank + li * nprocs;
            if (!beats(card[li], s, cur)) break;
            card[li] = generateCandidateCluster(s, clustered, points, grid, thr2, ws);
            dirty[li] = 0;
            if (beats(card[li], s, local)) local = {card[li], s};
            if (beats(card[li], s, cur)) cur = {card[li], s};
        }
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global.val <= 0) break;
        const int best_seed = global.loc;
        const int max_card = global.val;
        const int owner = best_seed % nprocs;

        members.resize(max_card);
        if (owner == rank) {
            generateCandidateCluster(best_seed, clustered, points, grid, thr2, ws,
                                     &members);
        }
        MPI_Bcast(members.data(), max_card, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = members;
        clusters.push_back(std::move(cluster));

        for (int p : members) clustered[p] = 1;
        remaining -= max_card;

        // Owned unclustered seeds that lose a neighbor become dirty
        for (int p : members) {
            my_grid.forNeighborCells(p, [&](int q) {
                if (clustered[q]) return;
                if (distance(points[q], points[p]) < threshold) {
                    const int li = q / nprocs;
                    nb[li]--;
                    card[li] = nb[li] + 1;
                    dirty[li] = 1;
                }
            });
        }
        alive.erase(std::remove_if(alive.begin(), alive.end(),
                                   [&](int li) { return clustered[rank + li * nprocs] != 0; }),
                    alive.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, const int rank) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Only rank 0 reports results
    if (rank != 0) return 0;
    
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 : 
        static_cast<double>(total_clustered) / clusters.size();
    
    printf("Points clustered: %d / %d (%.1f%%)\n", 
           total_clustered, num_points, 
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }
    
    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Suppress usage/error output on non-root ranks
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) { /* ignore */ }
    }
    const int rc = run(argc, argv, rank);
    fflush(stdout);
    MPI_Finalize();
    return rc;
}
