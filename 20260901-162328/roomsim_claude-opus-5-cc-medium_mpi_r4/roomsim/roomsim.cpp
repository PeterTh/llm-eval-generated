/**
 * Room Response Simulation Benchmark
 *
 * This is a distributed-memory (MPI) implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * MPI parallelization strategy
 * ----------------------------
 * The triangle index space is distributed over ranks as contiguous row blocks.
 * Mesh generation and the octree are replicated (cheap, and every rank needs to
 * trace rays against the whole scene).
 *
 *  - Form factors / time delays: each rank owns a contiguous block of rows i and
 *    computes Kij/Tau for all j. This is the dominant cost. The original code
 *    draws from one global mt19937 stream that is consumed strictly in row-major
 *    pair order, so every rank fast-forwards its engine to the exact stream
 *    position of its first pair (a pair consumes 4*NUM_RAYS draws unless it is
 *    culled by the normal test, which consumes none). Row blocks are sized so
 *    that each rank gets a similar number of non-culled pairs.
 *    Kij is folded into the weight the propagation needs and stored, together
 *    with Tau, for the local rows only, so the O(N^2) matrices are distributed
 *    instead of replicated.
 *  - Wave propagation: each rank updates radB for its own triangles and the
 *    timestep row is Allgathered so that all ranks have the full history needed
 *    for the delayed contributions of the next timestep.
 *  - Cross-correlation: distributed over triangles with an even split and
 *    Allgathered.
 *
 * Summation orders, the RNG stream and all floating point expressions are
 * identical to the sequential version, so results are bit-for-bit reproducible
 * and independent of the number of ranks.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
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

#include "../common/results_output.hpp"

// ============================================================================
// MPI helpers
// ============================================================================

static int mpiRank = 0;
static int mpiSize = 1;

// Only rank 0 produces output, so that the program's output matches the
// sequential version exactly.
__attribute__((format(printf, 1, 2)))
static void rprintf(const char* fmt, ...) {
    if (mpiRank != 0) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
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

    // Fast-forward the stream by n draws. The distribution consumes exactly one
    // engine value per rand() call, and mt19937::discard only performs the state
    // twists (no tempering), so this is essentially free compared to generating.
    void discard(uint64_t n) { rng.discard(n); }
};

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

// Cull triangles facing the same direction. Kept as a separate, non-inlined
// function so that the identical scalar code is used both when pre-counting the
// RNG draws of a row and when actually computing the form factor.
__attribute__((noinline))
bool isKijCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Number of RNG draws consumed by a non-culled pair: two barycentric points per
// ray, two uniform values per point.
constexpr uint64_t KIJ_RNG_DRAWS = 4 * static_cast<uint64_t>(NUM_RAYS);

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter).
// The caller has already established that the pair is not culled.
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

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
    std::vector<val_t> radE;        // Emission radiosity (T x local rows)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    // Coupling of the locally owned rows (localRows x N, row-major). The form
    // factor is pre-multiplied into the weight that the propagation loop needs;
    // pairs that do not couple carry a zero weight and therefore contribute
    // nothing, exactly like the skipped iterations of the sequential version.
    std::vector<val_t> weight;      // min(kij * area_j, 1), 0 if kij <= 0
    std::vector<int> tauv;          // time delay of the pair
    uint64_t nonZeroKij = 0;        // number of Kij entries > EPSILON (global)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Row block distribution (form factors + wave propagation)
    size_t rowBegin = 0, rowEnd = 0;
    std::vector<int> rowCounts, rowDispls;
    uint64_t rngOffset = 0;         // RNG stream position of the first local pair
    // Even distribution used by the cross-correlation phase
    size_t corrBegin = 0, corrEnd = 0;
    std::vector<int> corrCounts, corrDispls;

    size_t localRows() const { return rowEnd - rowBegin; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    size_t idxTLocal(size_t t, size_t n) const { return t * localRows() + (n - rowBegin); }
};

// Relative cost of fast-forwarding the RNG past one pair compared to actually
// computing that pair's form factor. Skipping a pair only twists the Mersenne
// state (~20 ns for 4*NUM_RAYS values) while computing one traces NUM_RAYS rays
// through the octree (tens of microseconds), hence the small ratio. It is a
// pure load balancing hint - the partition is correct for any value.
constexpr double RNG_SKIP_COST_RATIO = 0.0012;

// Split [0, n) into contiguous blocks, one per part, minimizing the maximum
// per-part cost of the form factor phase. Part p pays for the pairs it computes
// plus for fast-forwarding the RNG across all pairs that precede its block, so
// later parts get slightly smaller blocks. Every rank computes the same
// partition. Costs are expressed in units of "one computed pair".
static void partitionWeighted(const std::vector<uint64_t>& weights, int nparts,
                              std::vector<size_t>& bounds) {
    const size_t n = weights.size();
    std::vector<uint64_t> prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + weights[i];
    const double total = static_cast<double>(prefix[n]);

    // Greedily fill parts up to a makespan of 'limit'; true if all rows fit.
    auto feasible = [&](double limit, std::vector<size_t>* out) {
        size_t a = 0;
        for (int p = 0; p < nparts; ++p) {
            if (out) (*out)[static_cast<size_t>(p)] = a;
            if (a == n) continue;
            const double budget = limit - RNG_SKIP_COST_RATIO * static_cast<double>(prefix[a]);
            if (budget <= 0.0) return false;
            // Largest b with prefix[b] - prefix[a] <= budget
            const double cap = static_cast<double>(prefix[a]) + budget;
            size_t b = static_cast<size_t>(
                std::upper_bound(prefix.begin() + static_cast<long>(a) + 1, prefix.end(),
                                 static_cast<uint64_t>(cap)) - prefix.begin()) - 1;
            if (b <= a) b = a + 1;  // a row is the smallest unit of work
            a = std::min(b, n);
        }
        if (out) (*out)[static_cast<size_t>(nparts)] = n;
        return a == n;
    };

    bounds.assign(static_cast<size_t>(nparts) + 1, n);
    bounds[0] = 0;
    if (n == 0) return;

    double lo = total / nparts;                            // perfect balance
    double hi = total * (1.0 + RNG_SKIP_COST_RATIO);       // one part does all
    for (int it = 0; it < 64 && hi - lo > 1e-9 * hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (feasible(mid, nullptr)) hi = mid; else lo = mid;
    }
    if (!feasible(hi, &bounds)) {
        // Fall back to an even split of the weight (cannot normally happen)
        for (int p = 0; p <= nparts; ++p) {
            bounds[static_cast<size_t>(p)] = (n * static_cast<size_t>(p)) / static_cast<size_t>(nparts);
        }
    }
}

// Even contiguous split of [0, n) into nparts blocks.
static void partitionEven(size_t n, int nparts, std::vector<size_t>& bounds) {
    bounds.assign(static_cast<size_t>(nparts) + 1, 0);
    for (int p = 0; p <= nparts; ++p) {
        bounds[static_cast<size_t>(p)] =
            (n * static_cast<size_t>(p)) / static_cast<size_t>(nparts);
    }
}

// Turn block boundaries into MPI_Allgatherv counts/displacements.
static void boundsToCounts(const std::vector<size_t>& bounds, int nparts,
                           std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(static_cast<size_t>(nparts));
    displs.resize(static_cast<size_t>(nparts));
    for (int p = 0; p < nparts; ++p) {
        displs[static_cast<size_t>(p)] = static_cast<int>(bounds[static_cast<size_t>(p)]);
        counts[static_cast<size_t>(p)] =
            static_cast<int>(bounds[static_cast<size_t>(p) + 1] - bounds[static_cast<size_t>(p)]);
    }
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

    // Build octree for spatial acceleration (replicated: every rank traces rays
    // against the complete scene)
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Determine the row block owned by this rank. Blocks are balanced by the
    // number of non-culled pairs per row, which drives the cost of the form
    // factor phase, plus the cost of fast-forwarding the RNG to the block. The
    // counting pre-pass itself is distributed evenly.
    const size_t N = state.numTriangles;
    std::vector<size_t> preBounds;
    partitionEven(N, mpiSize, preBounds);

    std::vector<uint64_t> pairsPerRow(N, 0);
    for (size_t i = preBounds[static_cast<size_t>(mpiRank)];
         i < preBounds[static_cast<size_t>(mpiRank) + 1]; ++i) {
        uint64_t count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (!isKijCulled(state.triangles[i], state.triangles[j])) ++count;
        }
        pairsPerRow[i] = count;
    }
    MPI_Allreduce(MPI_IN_PLACE, pairsPerRow.data(), static_cast<int>(N),
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    std::vector<size_t> rowBounds;
    partitionWeighted(pairsPerRow, mpiSize, rowBounds);
    state.rowBegin = rowBounds[static_cast<size_t>(mpiRank)];
    state.rowEnd = rowBounds[static_cast<size_t>(mpiRank) + 1];
    boundsToCounts(rowBounds, mpiSize, state.rowCounts, state.rowDispls);

    // Stream position of the first pair of the local block: pairs are drawn in
    // row-major order and each non-culled pair consumes KIJ_RNG_DRAWS values.
    state.rngOffset = 0;
    for (size_t i = 0; i < state.rowBegin; ++i) state.rngOffset += pairsPerRow[i];
    state.rngOffset *= KIJ_RNG_DRAWS;

    std::vector<size_t> corrBounds;
    partitionEven(N, mpiSize, corrBounds);
    state.corrBegin = corrBounds[static_cast<size_t>(mpiRank)];
    state.corrEnd = corrBounds[static_cast<size_t>(mpiRank) + 1];
    boundsToCounts(corrBounds, mpiSize, state.corrCounts, state.corrDispls);

    // Initialize matrices (Kij/Tau are stored sparsely per local row)
    state.radE.assign(timesteps * state.localRows(), ZERO);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    if (state.sourceIndex >= state.rowBegin && state.sourceIndex < state.rowEnd) {
        for (size_t t = timeOn; t < timeOff; ++t) {
            state.radE[state.idxTLocal(t, state.sourceIndex)] = 1.0f;
        }
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Computes the form factors of the locally owned rows together with the
// matching time delays. Only this rank's rows are computed; the RNG is first
// advanced to the stream position of the block's first pair.
void computeFormFactors(SimulationState& state) {
    rprintf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    RandomGenerator rng(42);
    rng.discard(state.rngOffset);  // jump to this rank's position in the stream

    state.weight.assign(state.localRows() * N, ZERO);
    state.tauv.assign(state.localRows() * N, 0);

    uint64_t nonZero = 0;
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        const Triangle& triI = state.triangles[i];
        val_t* __restrict rowW = state.weight.data() + (i - state.rowBegin) * N;
        int* __restrict rowTau = state.tauv.data() + (i - state.rowBegin) * N;

        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const Triangle& triJ = state.triangles[j];
            // Culled pairs draw no random numbers and yield Kij == 0
            if (isKijCulled(triI, triJ)) continue;

            const val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
            if (kij > EPSILON) ++nonZero;
            if (kij <= ZERO) continue;

            rowTau[j] = computeTau(triI, triJ);
            rowW[j] = std::min(kij * state.areas[j], ONE);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &nonZero, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = nonZero;

    for (size_t i = 0; i < N; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            rprintf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    // Tau is computed together with the form factors for the pairs that
    // actually couple; see computeFormFactors().
    (void)state;
    rprintf("Computing time delays (Tau)...\n");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        val_t* __restrict radBt = state.radB.data() + t * N;
        const val_t* __restrict radBase = state.radB.data();

        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            const val_t* __restrict rowW = state.weight.data() + (i - state.rowBegin) * N;
            const int* __restrict rowTau = state.tauv.data() + (i - state.rowBegin) * N;
            val_t sumB = ZERO;

            // Same order and same terms as the sequential loop: pairs without a
            // coupling carry weight 0 and leave the sum unchanged.
            for (size_t j = 0; j < N; ++j) {
                const int tauij = rowTau[j];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                const size_t srcTime = t - static_cast<size_t>(tauij);
                const val_t radJ = radBase[srcTime * N + j];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += rowW[j] * radJ;
            }

            // Update radiosity: reflection + emission
            radBt[i] = state.rho[i] * sumB + state.radE[state.idxTLocal(t, i)];
        }

        // Publish this timestep's radiosities: every rank needs the full row to
        // evaluate the delayed contributions of the following timesteps.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, radBt,
                       state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

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

    const size_t T = state.numTimesteps;

    // Gather the (strided) time series once into contiguous buffers
    std::vector<val_t> src(T), sig(T);
    for (size_t t = 0; t < T; ++t) src[t] = state.radB[state.idxTN(t, state.sourceIndex)];

    for (size_t i = state.corrBegin; i < state.corrEnd; ++i) {
        for (size_t t = 0; t < T; ++t) sig[t] = state.radB[state.idxTN(t, i)];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                sum += src[tt - t] * sig[tt];
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(),
                   state.corrCounts.data(), state.corrDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
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

    // Check Kij matrix (should have some non-zero entries); counted while the
    // form factors were computed and reduced across ranks
    int nonZeroKij = static_cast<int>(state.nonZeroKij);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

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
    if (printResults && mpiRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (every rank holds the full distance/radiosity data, so all
    // ranks reach the same verdict)
    if (validate) {
        if (!validateResults(state)) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
