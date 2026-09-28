/**
 * Room Response Simulation Benchmark
 *
 * This is an MPI-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (MPI, distributed memory):
 *  - The mesh and the octree are cheap to build and are replicated on every rank.
 *  - The N x N Kij / Tau matrices are distributed over contiguous blocks of rows.
 *    Each rank only allocates and computes the rows it owns, so memory scales with
 *    1/P as well.
 *  - The random number stream of the sequential reference is reproduced bit-exactly:
 *    computeKij() consumes exactly NUM_RAYS*4 draws for every non-culled pair and
 *    none for culled pairs, so every rank can fast-forward its generator to the
 *    state the sequential code would have had at the start of its first row.
 *  - The row partition is balanced by the number of non-culled pairs per row
 *    (the cost driver of the form factor phase).
 *  - The wave propagation loop is parallel over rows; the radiosity vector of the
 *    current timestep is Allgathered so that all ranks can read the (delayed)
 *    radiosities of all triangles in later timesteps.
 *  - The cross correlation phase is parallel over rows, followed by one Allgather.
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
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// MPI helpers
// ============================================================================

static int g_rank = 0;
static int g_size = 1;

// printf that only produces output on the master rank, so that the program
// output is identical to the sequential reference implementation.
template <typename... Args>
static inline void rprintf(const char* fmt, Args... args) {
    if (g_rank == 0) {
        if constexpr (sizeof...(Args) == 0) {
            fputs(fmt, stdout);
        } else {
            printf(fmt, args...);
        }
    }
}

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
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

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
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
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
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

        // Collect all indices
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

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
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

        // Check if we can't split further (all triangles in one child)
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

        // Build children
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
    // Check if a ray intersects this node's bounding box
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

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
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

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
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

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
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

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
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

    // Fast-forward the stream by `n` calls to rand(). uniform_real_distribution
    // over float consumes exactly one 32 bit draw per call (24 mantissa bits),
    // and it carries no state, so this is equivalent to discarding n rand()s.
    void skip(uint64_t n) { rng.discard(n); }
};

// Number of rand() calls consumed by computeKij() for a pair that is not culled.
// (randomPointInTriangle draws 2 values and is called twice per ray.)
constexpr uint64_t RANDS_PER_PAIR = static_cast<uint64_t>(NUM_RAYS) * 4;

// Generate a random point inside a triangle using barycentric coordinates
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

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
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
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Cull triangles facing the same direction. Kept separate so that the number of
// random draws consumed by the form factor phase can be predicted without doing
// any of the expensive work (needed to reproduce the sequential RNG stream).
inline bool isCulledPair(const Vec3& normalI, const Vec3& normalJ) {
    return normalI.dot(normalJ) > 0.99f;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (isCulledPair(triI.normal(), triJ.normal())) return ZERO;

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

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (localRows x N matrix, row-major)
    std::vector<int> tau;           // Time delays (localRows x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, replicated)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // --- Row distribution (contiguous blocks of rows of the N x N matrices) ---
    size_t rowBegin = 0;            // First row owned by this rank
    size_t rowCount = 0;            // Number of rows owned by this rank
    std::vector<int> rowCounts;     // Per-rank row counts (for Allgatherv)
    std::vector<int> rowDispls;     // Per-rank row offsets (for Allgatherv)
    uint64_t rngSkip = 0;           // rand() calls consumed by the rows before rowBegin

    // Compressed coupling data for the propagation phase: for every local row
    // only the pairs with kij > 0 are stored, in ascending j order (which keeps
    // the floating point summation order of the sequential version).
    std::vector<size_t> rowPtr;     // rowCount + 1 offsets into the arrays below
    std::vector<idx_t> colIdx;      // Emitter triangle index j
    std::vector<int> colTau;        // Time delay of the pair
    std::vector<val_t> colWeight;   // min(kij * area_j, 1)

    size_t idx2d(size_t iLocal, size_t j) const { return iLocal * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Partition the rows into contiguous blocks with (approximately) the same amount
// of form factor work, measured as the number of non-culled pairs in a row.
// All ranks derive the identical partition from the same gathered row weights.
void partitionRows(SimulationState& state) {
    const size_t N = state.numTriangles;

    std::vector<Vec3> normals(N);
    for (size_t i = 0; i < N; ++i) normals[i] = state.triangles[i].normal();

    // Determining the weights is itself O(N^2), so it is done in parallel over an
    // even preliminary row split and the N results are gathered.
    std::vector<int> preCounts(static_cast<size_t>(g_size));
    std::vector<int> preDispls(static_cast<size_t>(g_size));
    {
        const size_t base = N / static_cast<size_t>(g_size);
        const size_t rem = N % static_cast<size_t>(g_size);
        size_t off = 0;
        for (int r = 0; r < g_size; ++r) {
            const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            preDispls[static_cast<size_t>(r)] = static_cast<int>(off);
            preCounts[static_cast<size_t>(r)] = static_cast<int>(cnt);
            off += cnt;
        }
    }

    std::vector<uint32_t> weights(N, 0);
    const size_t preBegin = static_cast<size_t>(preDispls[static_cast<size_t>(g_rank)]);
    const size_t preEnd = preBegin + static_cast<size_t>(preCounts[static_cast<size_t>(g_rank)]);
    for (size_t i = preBegin; i < preEnd; ++i) {
        uint32_t w = 0;
        const Vec3 ni = normals[i];
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (!isCulledPair(ni, normals[j])) ++w;
        }
        weights[i] = w;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, weights.data(), preCounts.data(),
                   preDispls.data(), MPI_UINT32_T, MPI_COMM_WORLD);

    // prefix[i] = number of non-culled pairs in the rows before i
    std::vector<uint64_t> prefix(N + 1, 0);
    for (size_t i = 0; i < N; ++i) prefix[i + 1] = prefix[i] + weights[i];
    const uint64_t total = prefix[N];

    // Contiguous split: rank r owns the rows whose prefix work crosses
    // [r*total/P, (r+1)*total/P). Every rank gets at most one extra row over the
    // ideal balance point, and empty ranks are possible only if P > N.
    std::vector<size_t> bounds(static_cast<size_t>(g_size) + 1, N);
    bounds[0] = 0;
    const bool oneRowPerRank = N >= static_cast<size_t>(g_size);
    size_t row = 0;
    for (int r = 1; r < g_size; ++r) {
        const uint64_t target = static_cast<uint64_t>(
            (static_cast<double>(total) * r) / g_size);
        while (row < N && prefix[row + 1] <= target &&
               (!oneRowPerRank || N - row > static_cast<size_t>(g_size - r))) {
            ++row;
        }
        bounds[static_cast<size_t>(r)] = row;
    }

    state.rowCounts.resize(static_cast<size_t>(g_size));
    state.rowDispls.resize(static_cast<size_t>(g_size));
    for (int r = 0; r < g_size; ++r) {
        state.rowDispls[static_cast<size_t>(r)] = static_cast<int>(bounds[static_cast<size_t>(r)]);
        state.rowCounts[static_cast<size_t>(r)] =
            static_cast<int>(bounds[static_cast<size_t>(r) + 1] - bounds[static_cast<size_t>(r)]);
    }

    state.rowBegin = bounds[static_cast<size_t>(g_rank)];
    state.rowCount = bounds[static_cast<size_t>(g_rank) + 1] - state.rowBegin;
    state.rngSkip = prefix[state.rowBegin] * RANDS_PER_PAIR;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration (replicated: cheap and deterministic)
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    // Distribute the rows of the N x N matrices over the ranks
    partitionRows(state);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (Kij / Tau: only the locally owned rows)
    state.kij.resize(state.rowCount * state.numTriangles, ZERO);
    state.tau.resize(state.rowCount * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
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
    rprintf("Computing form factors (Kij)...\n");

    // Fast-forward the generator to the state the sequential implementation
    // would have when it reaches this rank's first row.
    RandomGenerator rng(42);
    rng.skip(state.rngSkip);

    const size_t rowEnd = state.rowBegin + state.rowCount;
    for (size_t i = state.rowBegin; i < rowEnd; ++i) {
        val_t* kijRow = &state.kij[state.idx2d(i - state.rowBegin, 0)];
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            kijRow[j] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        if (g_rank == 0 && ((i + 1) % 100 == 0 || i + 1 == state.numTriangles)) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }

    // The remaining progress lines belong to rows owned by other ranks; emit them
    // once the whole phase has completed so that the output stays unchanged.
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = rowEnd; g_rank == 0 && i < state.numTriangles; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");

    const size_t rowEnd = state.rowBegin + state.rowCount;
    for (size_t i = state.rowBegin; i < rowEnd; ++i) {
        int* tauRow = &state.tau[state.idx2d(i - state.rowBegin, 0)];
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            tauRow[j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// Build the compressed coupling structure used by the propagation loop. This
// hoists the (timestep independent) min(kij * area_j, 1) product out of the
// innermost loop and drops the pairs that cannot contribute at all.
void buildCouplings(SimulationState& state) {
    const size_t N = state.numTriangles;

    state.rowPtr.assign(state.rowCount + 1, 0);
    size_t nnz = 0;
    for (size_t il = 0; il < state.rowCount; ++il) {
        const val_t* kijRow = &state.kij[state.idx2d(il, 0)];
        for (size_t j = 0; j < N; ++j) {
            if (kijRow[j] > ZERO) ++nnz;
        }
        state.rowPtr[il + 1] = nnz;
    }

    state.colIdx.resize(nnz);
    state.colTau.resize(nnz);
    state.colWeight.resize(nnz);

    for (size_t il = 0; il < state.rowCount; ++il) {
        const val_t* kijRow = &state.kij[state.idx2d(il, 0)];
        const int* tauRow = &state.tau[state.idx2d(il, 0)];
        size_t pos = state.rowPtr[il];
        for (size_t j = 0; j < N; ++j) {
            if (kijRow[j] <= ZERO) continue;
            state.colIdx[pos] = static_cast<idx_t>(j);
            state.colTau[pos] = tauRow[j];
            state.colWeight[pos] = std::min(kijRow[j] * state.areas[j], ONE);
            ++pos;
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    buildCouplings(state);

    const size_t N = state.numTriangles;
    const val_t* radB = state.radB.data();

    // Tau is ceil(distance / wave speed) between two distinct triangle centers and
    // therefore always >= 1, so a timestep only reads radiosities of strictly
    // earlier timesteps, which are already complete on every rank.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);

        for (size_t il = 0; il < state.rowCount; ++il) {
            const size_t i = state.rowBegin + il;
            const size_t end = state.rowPtr[il + 1];
            val_t sumB = ZERO;

            for (size_t p = state.rowPtr[il]; p < end; ++p) {
                const int tauij = state.colTau[p];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                const size_t srcTime = t - static_cast<size_t>(tauij);
                const val_t radJ = radB[srcTime * N + state.colIdx[p]];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += state.colWeight[p] * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Publish this timestep's radiosities: later timesteps read the delayed
        // radiosity of every triangle, so all ranks need the full vector.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       &state.radB[state.idxTN(t, 0)], state.rowCounts.data(),
                       state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            rprintf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    rprintf("Computing distances via cross-correlation...\n");

    for (size_t i = state.rowBegin; i < state.rowBegin + state.rowCount; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    // Collect the distances of all triangles on every rank
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    rprintf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            rprintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rprintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rprintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rprintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rprintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rprintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    rprintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rprintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries); it is row-distributed,
    // so the local counts have to be summed up over all ranks.
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.kij.size(); ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    MPI_Allreduce(MPI_IN_PLACE, &nonZeroKij, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    rprintf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        rprintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rprintf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
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
    rprintf("Usage: %s [options]\n", progName);
    rprintf("Options:\n");
    rprintf("  -n <num>     Target number of triangles (default: 320)\n");
    rprintf("               Actual count will be rounded to nearest icosphere level:\n");
    rprintf("               20, 80, 320, 1280, 5120, 20480\n");
    rprintf("  -t <num>     Number of timesteps (default: 50)\n");
    rprintf("  -s <num>     Source triangle index (default: 0)\n");
    rprintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rprintf("  -v           Enable validation\n");
    rprintf("  -o           Print results for external validation\n");
    rprintf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    static_assert(std::is_same_v<val_t, float>, "MPI_FLOAT must match val_t");

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            rprintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rprintf("Room Response Simulation Benchmark\n");
    rprintf("===================================\n");
    rprintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rprintf("Timesteps: %d\n", timesteps);
    rprintf("Source triangle: %d\n", sourceIdx);
    rprintf("Reflectivity: %.2f\n", reflectivity);
    rprintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rprintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rprintf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rprintf("Distance computation time: %ld ms\n", distDuration);
    rprintf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    rprintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    rprintf("\nPerformance:\n");
    rprintf("  Triangles: %zu\n", n);
    rprintf("  Timesteps: %zu\n", t);
    rprintf("  Form factor computations: %.2e\n", kijOps);
    rprintf("  Simulation operations: %.2e\n", simOps);
    rprintf("  Distance computations: %.2e\n", distOps);
    rprintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    rprintf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    rprintf("  Result hash: %016lX\n", hash);
    rprintf("\n");
    
    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (collective: the Kij matrix is distributed)
    const bool ok = validate ? validateResults(state) : true;

    // Make sure the master rank's output has left the process before any rank
    // exits, otherwise a failing run may be torn down mid-print by the launcher.
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
