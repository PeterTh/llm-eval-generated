/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
#include <sstream>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static int g_rank = 0;
static int g_size = 1;

#define PRINT0(...) do { if (g_rank == 0) std::printf(__VA_ARGS__); } while (0)

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

    void discard(uint64_t n) { rng.discard(n); }

    std::string state_string() const {
        std::ostringstream os;
        os << rng;
        return os.str();
    }

    void set_state_string(const std::string& s) {
        std::istringstream is(s);
        is >> rng;
    }
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

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
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

static inline void block_decompose(size_t n, int size, int rank, size_t& begin, size_t& count) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    count = base + (static_cast<size_t>(rank) < rem ? 1u : 0u);
    begin = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
}

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // replicated (N)
    std::vector<val_t> rho;         // replicated (N)

    // Distributed ownership over triangle index i (contiguous block)
    size_t localBegin = 0;
    size_t localCount = 0;
    std::vector<int> counts;        // size = g_size
    std::vector<int> displs;        // size = g_size

    // Distributed matrices (local rows only): shape (localCount x N)
    std::vector<val_t> kijLocal;
    std::vector<int> tauLocal;

    // Radiosity stored for local triangles only: shape (T x localCount)
    std::vector<val_t> radBLocal;

    // Global radiosity ring buffer for time-delay lookups: shape ((maxTau+1) x N)
    size_t maxTau = 0;
    std::vector<val_t> radBGlobalRing;

    // Distances: local and (rank0) gathered global
    std::vector<val_t> distancesLocal;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex = 0;
    size_t timeOff = 0;

    size_t idx2dLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTNLocal(size_t t, size_t li) const { return t * localCount + li; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = state.numTriangles ? (sourceIdx % state.numTriangles) : 0;
    state.timeOff = timesteps / 2;

    PRINT0("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    PRINT0("Building octree...\n");
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) state.areas[i] = state.triangles[i].area();

    state.rho.resize(state.numTriangles, reflectivity);

    block_decompose(state.numTriangles, g_size, g_rank, state.localBegin, state.localCount);

    state.counts.assign(g_size, 0);
    state.displs.assign(g_size, 0);
    for (int r = 0; r < g_size; ++r) {
        size_t b = 0, c = 0;
        block_decompose(state.numTriangles, g_size, r, b, c);
        state.counts[r] = static_cast<int>(c);
        state.displs[r] = static_cast<int>(b);
    }

    state.kijLocal.assign(state.localCount * state.numTriangles, ZERO);
    state.tauLocal.assign(state.localCount * state.numTriangles, 0);
    state.radBLocal.assign(state.numTimesteps * state.localCount, ZERO);
    state.distancesLocal.assign(state.localCount, ZERO);
    if (g_rank == 0) state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    PRINT0("Computing form factors (Kij) with %d MPI ranks...\n", g_size);

    // Preserve the exact sequential RNG stream by initializing each rank's RNG
    // to the stream position corresponding to its first owned i-row.
    std::string myState;
    {
        const uint64_t drawsPerPair = static_cast<uint64_t>(4u * NUM_RAYS);
        const uint64_t drawsPerRow = static_cast<uint64_t>(state.numTriangles ? (state.numTriangles - 1) : 0) * drawsPerPair;

        int myLen = 0;
        if (g_size == 1) {
            RandomGenerator rg(42);
            myState = rg.state_string();
        } else {
            std::vector<int> lens;
            std::vector<int> charDispls;
            std::vector<char> allChars;

            if (g_rank == 0) {
                lens.assign(g_size, 0);
                std::vector<std::string> states(static_cast<size_t>(g_size));

                RandomGenerator rg(42);
                size_t currentRow = 0;
                for (int r = 0; r < g_size; ++r) {
                    const size_t beginRow = static_cast<size_t>(state.displs[r]);
                    const uint64_t deltaRows = static_cast<uint64_t>(beginRow - currentRow);
                    rg.discard(deltaRows * drawsPerRow);
                    states[static_cast<size_t>(r)] = rg.state_string();
                    lens[r] = static_cast<int>(states[static_cast<size_t>(r)].size());
                    currentRow = beginRow;
                }

                charDispls.assign(g_size, 0);
                int total = 0;
                for (int r = 0; r < g_size; ++r) {
                    charDispls[r] = total;
                    total += lens[r];
                }
                allChars.resize(static_cast<size_t>(total));
                for (int r = 0; r < g_size; ++r) {
                    const auto& s = states[static_cast<size_t>(r)];
                    std::memcpy(allChars.data() + charDispls[r], s.data(), static_cast<size_t>(lens[r]));
                }
            }

            MPI_Scatter(g_rank == 0 ? lens.data() : nullptr, 1, MPI_INT,
                        &myLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<char> myChars(static_cast<size_t>(myLen));
            MPI_Scatterv(g_rank == 0 ? allChars.data() : nullptr,
                         g_rank == 0 ? lens.data() : nullptr,
                         g_rank == 0 ? charDispls.data() : nullptr,
                         MPI_CHAR,
                         myChars.data(), myLen, MPI_CHAR,
                         0, MPI_COMM_WORLD);
            myState.assign(myChars.begin(), myChars.end());
        }
    }

    RandomGenerator rng(42);
    rng.set_state_string(myState);

    for (size_t li = 0; li < state.localCount; ++li) {
        const size_t i = state.localBegin + li;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kijLocal[state.idx2dLocal(li, j)] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        if (g_rank == 0 && (((i + 1) % 100) == 0 || (i + 1) == state.numTriangles)) {
            PRINT0("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    PRINT0("Computing time delays (Tau) with %d MPI ranks...\n", g_size);

    size_t localMaxTau = 0;
    for (size_t li = 0; li < state.localCount; ++li) {
        const size_t i = state.localBegin + li;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            const int tauij = computeTau(state.triangles[i], state.triangles[j]);
            state.tauLocal[state.idx2dLocal(li, j)] = tauij;
            localMaxTau = std::max(localMaxTau, static_cast<size_t>(tauij));
        }
    }

    unsigned long long l = static_cast<unsigned long long>(localMaxTau);
    unsigned long long g = 0;
    MPI_Allreduce(&l, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    state.maxTau = static_cast<size_t>(g);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    PRINT0("Running wave propagation simulation with %d MPI ranks...\n", g_size);

    const size_t bufLen = state.maxTau + 1;
    state.radBGlobalRing.assign(bufLen * state.numTriangles, ZERO);
    std::vector<val_t> globalRow(state.numTriangles, ZERO);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Compute local radiosity for owned triangles at time t (depends only on earlier timesteps).
        for (size_t li = 0; li < state.localCount; ++li) {
            const size_t i = state.localBegin + li;
            val_t sumB = ZERO;

            for (size_t j = 0; j < state.numTriangles; ++j) {
                if (i == j) continue;

                const int tauij = state.tauLocal[state.idx2dLocal(li, j)];
                if (static_cast<int>(t) < tauij) continue;

                const val_t kij = state.kijLocal[state.idx2dLocal(li, j)];
                if (kij <= ZERO) continue;

                const size_t srcTime = t - static_cast<size_t>(tauij);
                const val_t radJ = state.radBGlobalRing[(srcTime % bufLen) * state.numTriangles + j];
                if (radJ <= ZERO) continue;

                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            const val_t radE = (i == state.sourceIndex && t < state.timeOff) ? ONE : ZERO;
            state.radBLocal[state.idxTNLocal(t, li)] = state.rho[i] * sumB + radE;
        }

        // Publish global radiosity row t for time-delay lookups.
        MPI_Allgatherv(state.localCount ? (state.radBLocal.data() + t * state.localCount) : nullptr,
                       static_cast<int>(state.localCount), MPI_FLOAT,
                       globalRow.data(), state.counts.data(), state.displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
        std::memcpy(state.radBGlobalRing.data() + (t % bufLen) * state.numTriangles,
                    globalRow.data(), state.numTriangles * sizeof(val_t));

        if (g_rank == 0 && (((t + 1) % 10) == 0 || (t + 1) == state.numTimesteps)) {
            PRINT0("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    PRINT0("Computing distances via cross-correlation with %d MPI ranks...\n", g_size);

    // Broadcast the source radiosity time series to all ranks.
    int sourceOwner = 0;
    for (int r = 0; r < g_size; ++r) {
        const int b = state.displs[r];
        const int e = b + state.counts[r];
        if (static_cast<int>(state.sourceIndex) >= b && static_cast<int>(state.sourceIndex) < e) {
            sourceOwner = r;
            break;
        }
    }

    std::vector<val_t> sourceSeries(state.numTimesteps, ZERO);
    if (g_rank == sourceOwner && state.localCount) {
        const size_t liSrc = state.sourceIndex - state.localBegin;
        for (size_t tt = 0; tt < state.numTimesteps; ++tt) {
            sourceSeries[tt] = state.radBLocal[state.idxTNLocal(tt, liSrc)];
        }
    }
    MPI_Bcast(sourceSeries.data(), static_cast<int>(state.numTimesteps), MPI_FLOAT, sourceOwner, MPI_COMM_WORLD);

    for (size_t li = 0; li < state.localCount; ++li) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        for (size_t lag = 0; lag < state.numTimesteps; ++lag) {
            val_t sum = ZERO;
            for (size_t tt = lag; tt < state.numTimesteps; ++tt) {
                const val_t pB = state.radBLocal[state.idxTNLocal(tt, li)];
                const val_t pS = sourceSeries[tt - lag];
                sum += pS * pB;
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(lag);
            }
        }

        state.distancesLocal[li] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Gatherv(state.localCount ? state.distancesLocal.data() : nullptr,
                static_cast<int>(state.localCount), MPI_FLOAT,
                g_rank == 0 ? state.distances.data() : nullptr,
                state.counts.data(), state.displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResultsMPI(const SimulationState& state) {
    // Reduce distributed validation stats.
    long long localReceived = 0;
    for (size_t li = 0; li < state.localCount; ++li) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radBLocal[state.idxTNLocal(t, li)] > EPSILON) {
                localReceived++;
                break;
            }
        }
    }

    long long localNonZeroKij = 0;
    for (const auto& v : state.kijLocal) {
        if (v > EPSILON) localNonZeroKij++;
    }

    long long receivedEnergy = 0;
    long long nonZeroKij = 0;
    MPI_Reduce(&localReceived, &receivedEnergy, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    bool ok = true;
    if (g_rank == 0) {
        std::printf("\nValidation:\n");

        bool allNonNegative = true;
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;

        for (size_t i = 0; i < state.numTriangles; ++i) {
            const val_t d = state.distances[i];
            if (d < 0) {
                allNonNegative = false;
                std::printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
            }
            if (!std::isfinite(d)) {
                std::printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
                ok = false;
                break;
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) nonZeroCount++;
        }

        if (ok) {
            std::printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
            std::printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
            std::printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

            const val_t srcDist = state.distances[state.sourceIndex];
            if (srcDist > WAVE_SPEED * 2) {
                std::printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
            }

            std::printf("  Triangles receiving energy: %lld/%zu\n", receivedEnergy, state.numTriangles);
            if (receivedEnergy == 0) {
                std::printf("  ERROR: No triangles received energy - simulation failed\n");
                ok = false;
            }

            const unsigned long long nn = static_cast<unsigned long long>(state.numTriangles) * static_cast<unsigned long long>(state.numTriangles);
            std::printf("  Non-zero form factors: %lld/%llu (%.2f%%)\n",
                        nonZeroKij, nn,
                        100.0f * static_cast<double>(nonZeroKij) / static_cast<double>(nn));
            if (nonZeroKij == 0) {
                std::printf("  ERROR: All form factors are zero - visibility computation failed\n");
                ok = false;
            }

            if (!allNonNegative) ok = false;
        }

        std::printf("  Validation: %s\n", ok ? "PASSED" : "FAILED");
    }

    int okInt = ok ? 1 : 0;
    MPI_Bcast(&okInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return okInt == 1;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&distances[i]);
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
    PRINT0("Usage: %s [options]\n", progName);
    PRINT0("Options:\n");
    PRINT0("  -n <num>     Target number of triangles (default: 320)\n");
    PRINT0("               Actual count will be rounded to nearest icosphere level:\n");
    PRINT0("               20, 80, 320, 1280, 5120, 20480\n");
    PRINT0("  -t <num>     Number of timesteps (default: 50)\n");
    PRINT0("  -s <num>     Source triangle index (default: 0)\n");
    PRINT0("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    PRINT0("  -v           Enable validation\n");
    PRINT0("  -o           Print results for external validation\n");
    PRINT0("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    int exitCode = 0;

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
    bool doRun = true;

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
            doRun = false;
            break;
        } else {
            PRINT0("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            exitCode = 1;
            doRun = false;
            break;
        }
    }

    if (!doRun) {
        MPI_Finalize();
        return exitCode;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    PRINT0("Room Response Simulation Benchmark\n");
    PRINT0("===================================\n");
    PRINT0("MPI ranks: %d\n", g_size);
    PRINT0("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    PRINT0("Timesteps: %d\n", timesteps);
    PRINT0("Source triangle: %d\n", sourceIdx);
    PRINT0("Reflectivity: %.2f\n", reflectivity);
    PRINT0("Validation: %s\n\n", validate ? "enabled" : "disabled");

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    PRINT0("\n");

    long preMaxMs = 0, simMaxMs = 0, distMaxMs = 0;

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();
    computeTimeDelays(state);
    computeFormFactors(state);
    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    long preLocal = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    MPI_Reduce(&preLocal, &preMaxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    long simLocal = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    MPI_Reduce(&simLocal, &simMaxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    long distLocal = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    MPI_Reduce(&distLocal, &distMaxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::printf("Precomputation time: %ld ms\n\n", preMaxMs);
        std::printf("Simulation time: %ld ms\n\n", simMaxMs);
        std::printf("Distance computation time: %ld ms\n\n", distMaxMs);

        const long totalTime = preMaxMs + simMaxMs + distMaxMs;
        std::printf("Total computation time: %ld ms\n", totalTime);

        const size_t n = state.numTriangles;
        const size_t T = state.numTimesteps;
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(T);
        const double distOps = static_cast<double>(n) * static_cast<double>(T) * static_cast<double>(T);

        std::printf("\nPerformance:\n");
        std::printf("  Triangles: %zu\n", n);
        std::printf("  Timesteps: %zu\n", T);
        std::printf("  Form factor computations: %.2e\n", kijOps);
        std::printf("  Simulation operations: %.2e\n", simOps);
        std::printf("  Distance computations: %.2e\n", distOps);
        std::printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / static_cast<double>(n));

        const size_t memLocal = state.kijLocal.size() * sizeof(val_t) +
                                state.tauLocal.size() * sizeof(int) +
                                state.radBLocal.size() * sizeof(val_t) +
                                state.radBGlobalRing.size() * sizeof(val_t);
        std::printf("  Approx. local memory: %.2f MB\n", memLocal / (1024.0 * 1024.0));

        const uint64_t hash = computeHash(state.distances);
        std::printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    if (validate) {
        if (!validateResultsMPI(state)) exitCode = 1;
    }

    MPI_Finalize();
    return exitCode;
}
