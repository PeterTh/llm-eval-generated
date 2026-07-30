/**
 * Room Response Simulation Benchmark - Hybrid MPI+OpenMP+CUDA
 * 
 * Hybrid parallel implementation combining:
 * - MPI: Distribution of triangle rows across nodes/processes
 * - OpenMP: CPU thread parallelism for form factors (with octree visibility)
 * - CUDA: GPU acceleration for wave propagation and distance computation
 * 
 * The form factor computation uses octree-accelerated visibility on CPU (OpenMP),
 * while the regular grid computations (wave propagation, cross-correlation) 
 * run on GPU (CUDA). MPI distributes work across processes.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }

public:
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;
        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t rowStart, rowEnd;  // MPI: local row range [rowStart, rowEnd)

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;         // Local rows: (rowEnd-rowStart) x N
    std::vector<int> tau;           // Local rows: (rowEnd-rowStart) x N
    std::vector<val_t> radE;        // T x N (full, replicated)
    std::vector<val_t> radB;        // T x N (full, replicated)
    std::vector<val_t> distances;   // Local only: (rowEnd-rowStart)

    Octree octree;
    size_t sourceIndex;

    int mpiRank, mpiSize;

    size_t idx2d(size_t i, size_t j) const { return (i - rowStart) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Helper: compute row distribution
static inline void getRowDistribution(size_t N, int size, int rank, size_t& start, size_t& end) {
    size_t rowsPerRank = N / size;
    size_t remainder = N % size;
    start = rank * rowsPerRank + std::min<size_t>(rank, remainder);
    end = start + rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int size) {
    state.mpiRank = rank;
    state.mpiSize = size;

    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    getRowDistribution(state.numTriangles, size, rank, state.rowStart, state.rowEnd);

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(state.numTriangles, reflectivity);

    size_t localRows = state.rowEnd - state.rowStart;
    state.kij.resize(localRows * state.numTriangles, ZERO);
    state.tau.resize(localRows * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(localRows, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    size_t N = state.numTriangles;
    size_t localRows = state.rowEnd - state.rowStart;

    // Pre-generate all random values sequentially to match original RNG sequence.
    // The above approach has a bug in random value indexing. Let me use a simpler approach:
    // Just use per-thread RNG with deterministic seeding to reproduce results.
    // Actually, the simplest correct approach: use the original sequential computation
    // but parallelize with OpenMP where each thread handles a range of rows with its
    // own RNG that's seeded to produce the same values as the global sequential RNG.
    
    // Reset and use direct sequential-per-row approach with OpenMP
    // Each row i consumes N * NUM_RAYS * 4 random values (for all j's in that row)
    // Actually no - the original uses a SINGLE rng across ALL (i,j) pairs sequentially.
    // Row i consumes: sum over j of (NUM_RAYS * 2) values = N * NUM_RAYS * 2 values
    // Wait, computeKij calls randomPointInTriangle twice per ray, each consuming 2 values
    // So per (i,j): NUM_RAYS * 4 values? No:
    // randomPointInTriangle consumes 2 values (u, v)
    // computeKij calls it twice per ray: once for pI, once for pJ
    // So per ray: 4 values. Per (i,j) pair: NUM_RAYS * 4 values.
    
    // Actually let me re-read the original:
    // for (int r = 0; r < NUM_RAYS; ++r) {
    //     Vec3 pI = randomPointInTriangle(triI, rng);  // 2 rand() calls
    //     Vec3 pJ = randomPointInTriangle(triJ, rng);  // 2 rand() calls
    // }
    // So per (i,j): NUM_RAYS * 4 values = 64 values
    
    // For parallelization, we need to know how many values each row consumes:
    // Row i: N * NUM_RAYS * 4 values (for j=0..N-1, skipping i==j still consumes values? 
    //         NO - original has "if (i == j) continue" BEFORE computeKij)
    // Actually the original: for j, if i==j continue, else computeKij
    // So row i consumes: (N-1) * NUM_RAYS * 4 values (skipping j==i)
    // But the j==i case doesn't call computeKij so doesn't consume RNG values.
    // This means the RNG offset for row i depends on how many j<i cases were skipped...
    // Actually no, the "continue" for i==j happens before computeKij, so no RNG consumed.
    // For row i, the RNG values consumed for j=0..i-1 is i * NUM_RAYS * 4
    // For j=i: 0 values (continue)
    // For j=i+1..N-1: (N-1-i) * NUM_RAYS * 4 values
    // Total for row i: (N-1) * NUM_RAYS * 4 values
    
    // Total before row i: i * (N-1) * NUM_RAYS * 4 values
    // This is correct because each row consumes exactly (N-1) pairs worth of RNG.
    
    // OK let me just redo this properly.
    // Actually, the simplest correct approach for OpenMP: 
    // Each thread creates its own RNG, seeded to match the global sequence at its starting row.
    
    size_t randsPerPair = NUM_RAYS * 4; // 64
    size_t pairsPerRow = N - 1; // skip i==j
    size_t randsPerRow = pairsPerRow * randsPerPair;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t localI = 0; localI < static_cast<int64_t>(localRows); ++localI) {
        size_t i = state.rowStart + localI;
        
        // Create RNG for this row, seeded to match global sequence
        RandomGenerator rng(42);
        // Skip all values consumed by rows 0..(i-1)
        // Each row k consumes (N-1) * randsPerPair values
        size_t totalSkip = static_cast<size_t>(i) * randsPerRow;
        for (size_t s = 0; s < totalSkip; ++s) {
            rng.rand();
        }
        
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                state.kij[localI * N + j] = ZERO;
                continue;
            }
            
            state.kij[localI * N + j] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            #pragma omp critical
            {
                printf("  Progress: %zu/%zu triangles\n", i + 1, N);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;
    size_t localRows = state.rowEnd - state.rowStart;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t localI = 0; localI < static_cast<int64_t>(localRows); ++localI) {
        size_t i = state.rowStart + localI;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                state.tau[localI * N + j] = 0;
                continue;
            }
            state.tau[localI * N + j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// CUDA Kernels for Wave Propagation and Distance Computation
// ============================================================================

// CUDA kernel: wave propagation for one timestep
// Each thread computes radB[t][i] for one local row i
__global__ void wavePropagationKernel(
    const val_t* kij,
    const int* tau,
    const val_t* areas,
    const val_t* rho,
    const val_t* radE,
    const val_t* radB,
    val_t* radBNew,
    size_t N,
    size_t t,
    size_t rowStart,
    size_t rowEnd)
{
    size_t i = rowStart + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowEnd) return;

    size_t localI = i - rowStart;
    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[localI * N + j];
        if (static_cast<int>(t) < tauij) continue;

        val_t kijVal = kij[localI * N + j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijVal * areas[j], ONE) * radJ;
    }

    radBNew[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

// CUDA kernel: distance computation via cross-correlation
__global__ void computeDistancesKernel(
    const val_t* __restrict__ radB,
    val_t* __restrict__ distances,
    size_t N,
    size_t numTimesteps,
    size_t sourceIndex,
    size_t rowStart,
    size_t rowEnd)
{
    size_t i = rowStart + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowEnd) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;

        for (size_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i - rowStart] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Simulation Phase (Wave Propagation) - CUDA + MPI
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    size_t localRows = state.rowEnd - state.rowStart;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Allocate GPU memory
    val_t *d_kij, *d_areas, *d_rho, *d_radE, *d_radB;
    int* d_tau;

    CUDA_CHECK(cudaMalloc(&d_kij, localRows * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_tau, localRows * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_radE, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(val_t)));

    // Upload data
    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), localRows * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), localRows * N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    // Precompute MPI distribution info
    std::vector<int> counts(state.mpiSize), displs(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t rs, re;
        getRowDistribution(N, state.mpiSize, r, rs, re);
        counts[r] = static_cast<int>(re - rs);
        displs[r] = static_cast<int>(rs);
    }

    dim3 blockSize(256);
    dim3 gridSize((localRows + blockSize.x - 1) / blockSize.x);

    std::vector<val_t> localRadB(localRows);
    std::vector<val_t> allRadB(N);

    for (size_t t = 0; t < T; ++t) {
        // Run wave propagation on GPU
        wavePropagationKernel<<<gridSize, blockSize>>>(
            d_kij, d_tau, d_areas, d_rho, d_radE, d_radB, d_radB,
            N, t, state.rowStart, state.rowEnd);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Extract local rows from GPU
        CUDA_CHECK(cudaMemcpy(localRadB.data(), d_radB + t * N + state.rowStart,
                              localRows * sizeof(val_t), cudaMemcpyDeviceToHost));

        // MPI: Allgatherv to share radB across all ranks
        MPI_Allgatherv(localRadB.data(), static_cast<int>(localRows), MPI_FLOAT,
                       allRadB.data(), counts.data(), displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Upload full radB row to GPU
        CUDA_CHECK(cudaMemcpy(d_radB + t * N, allRadB.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            if (state.mpiRank == 0) {
                printf("  Timestep %zu/%zu\n", t + 1, T);
            }
        }
    }

    // Download full radB
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, T * N * sizeof(val_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
}

// ============================================================================
// Distance Computation (Cross-Correlation) - CUDA + MPI
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    size_t localRows = state.rowEnd - state.rowStart;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    val_t* d_radB;
    val_t* d_distances;

    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, localRows * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 gridSize((localRows + blockSize.x - 1) / blockSize.x);

    computeDistancesKernel<<<gridSize, blockSize>>>(
        d_radB, d_distances, N, T, state.sourceIndex, state.rowStart, state.rowEnd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, localRows * sizeof(val_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_distances));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    // Gather all distances to rank 0
    std::vector<val_t> allDistances;
    std::vector<int> counts(state.mpiSize), displs(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t rs, re;
        getRowDistribution(state.numTriangles, state.mpiSize, r, rs, re);
        counts[r] = static_cast<int>(re - rs);
        displs[r] = static_cast<int>(rs);
    }

    if (state.mpiRank == 0) {
        allDistances.resize(state.numTriangles);
    }

    size_t localRows = state.rowEnd - state.rowStart;
    MPI_Gatherv(state.distances.data(), static_cast<int>(localRows), MPI_FLOAT,
                allDistances.data(), counts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (state.mpiRank != 0) return true;

    printf("\nValidation:\n");

    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = allDistances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    val_t srcDist = allDistances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity - gather from all ranks
    // For simplicity, check locally on rank 0 with its own rows
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    // MPI reduce to get total
    int globalReceivedEnergy = 0;
    MPI_Reduce(&receivedEnergy, &globalReceivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("  Triangles receiving energy: %d/%zu\n", globalReceivedEnergy, state.numTriangles);

    if (globalReceivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix
    int localNonZeroKij = 0;
    for (size_t i = 0; i < localRows * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }

    int globalNonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    size_t totalKijEntries = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           globalNonZeroKij, totalKijEntries,
           100.0f * globalNonZeroKij / static_cast<val_t>(totalKijEntries));

    if (globalNonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    std::vector<val_t> allDistances;
    std::vector<int> counts(state.mpiSize), displs(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t rs, re;
        getRowDistribution(state.numTriangles, state.mpiSize, r, rs, re);
        counts[r] = static_cast<int>(re - rs);
        displs[r] = static_cast<int>(rs);
    }

    if (state.mpiRank == 0) {
        allDistances.resize(state.numTriangles);
    }

    size_t localRows = state.rowEnd - state.rowStart;
    MPI_Gatherv(state.distances.data(), static_cast<int>(localRows), MPI_FLOAT,
                allDistances.data(), counts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (state.mpiRank != 0) return 0;

    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&allDistances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Select GPU based on MPI rank
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    if (numGPUs > 0) {
        int gpuId = rank % numGPUs;
        CUDA_CHECK(cudaSetDevice(gpuId));
    }

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
    }

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, size);

    if (rank == 0) printf("\n");

    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    long totalTime = preDuration + simDuration + distDuration;
    if (rank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);

        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
    }

    uint64_t hash = computeHash(state);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printResults) {
        // Gather all distances to rank 0
        std::vector<val_t> allDistances;
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rs, re;
            getRowDistribution(state.numTriangles, size, r, rs, re);
            counts[r] = static_cast<int>(re - rs);
            displs[r] = static_cast<int>(rs);
        }

        if (rank == 0) allDistances.resize(state.numTriangles);

        size_t localRows = state.rowEnd - state.rowStart;
        MPI_Gatherv(state.distances.data(), static_cast<int>(localRows), MPI_FLOAT,
                    allDistances.data(), counts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<double> distData(allDistances.begin(), allDistances.end());
            print_results(distData, "Distances");
        }
    }

    if (validate) {
        if (!validateResults(state)) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
