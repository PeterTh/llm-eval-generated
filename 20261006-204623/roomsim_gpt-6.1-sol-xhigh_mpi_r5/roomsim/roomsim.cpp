/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed-memory MPI implementation of room impulse response
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

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
    // A float uniform sample consumes one 32-bit mt19937 output. Discarding
    // preserves the serial stream without evaluating another rank's rays.
    void discard(uint64_t draws) { rng.discard(draws); }
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

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t firstTriangle;
    size_t localTriangles;
    int rank;
    int numRanks;
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<int> rowCounts, rowDisplacements;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;
    std::vector<val_t> kij;         // Area-weighted factors (local rows x N)
    std::vector<uint8_t> tau;       // Delays (local rows x N); radius 10 gives <= 40
    std::vector<val_t> radB;        // Local reflected radiosity (T x local rows)
    std::vector<val_t> history;     // Global radiosity in a circular delay buffer
    std::vector<val_t> sourceHistory;
    std::vector<val_t> exchangeBuffer;
    std::vector<val_t> distances;   // Local distances, gathered on rank zero
    uint64_t eligiblePairs = 0;
    uint64_t nonZeroKij = 0;
    size_t maxDelay = 0;
    size_t exchangeBatch = 1;
    size_t historyLength = 1;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * localTriangles + n; }

    ~SimulationState() {
        // Let main abort the entire job immediately if a rank throws.
        if (comm != MPI_COMM_NULL && std::uncaught_exceptions() == 0) MPI_Comm_free(&comm);
    }
};

// ============================================================================
// Initialization
// ============================================================================

bool initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // Excess ranks participate in MPI finalization, but need no simulation data
    // or timestep collectives when there are fewer receiver rows than ranks.
    MPI_Comm_split(MPI_COMM_WORLD,
                   static_cast<size_t>(worldRank) < state.numTriangles ? 0 : MPI_UNDEFINED,
                   worldRank, &state.comm);
    if (state.comm == MPI_COMM_NULL) return false;
    MPI_Comm_rank(state.comm, &state.rank);
    MPI_Comm_size(state.comm, &state.numRanks);
    state.rowCounts.resize(state.numRanks);
    state.rowDisplacements.resize(state.numRanks);
    for (int r = 0; r < state.numRanks; ++r) {
        size_t first = state.numTriangles * static_cast<size_t>(r) / state.numRanks;
        size_t last = state.numTriangles * static_cast<size_t>(r + 1) / state.numRanks;
        state.rowCounts[r] = static_cast<int>(last - first);
        state.rowDisplacements[r] = static_cast<int>(first);
    }
    state.firstTriangle = static_cast<size_t>(state.rowDisplacements[state.rank]);
    state.localTriangles = static_cast<size_t>(state.rowCounts[state.rank]);

    if (state.rank == 0) {
        printf("MPI processes: %d (%d active)\n", worldSize, state.numRanks);
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Only receiver rows and their full time histories are distributed. The
    // source emission is implicit, avoiding a separate, mostly zero matrix.
    state.kij.resize(state.localTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.localTriangles * state.numTriangles, 0);
    state.radB.resize(timesteps * state.localTriangles, ZERO);
    state.sourceHistory.resize(timesteps, ZERO);
    state.distances.resize(state.localTriangles, ZERO);
    return true;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    uint64_t precedingPairs = 0;
    MPI_Exscan(&state.eligiblePairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, state.comm);
    if (state.rank == 0) precedingPairs = 0;
    RandomGenerator rng(42);
    rng.discard(precedingPairs * (4 * NUM_RAYS));

    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        size_t i = state.firstTriangle + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            val_t factor = computeKij(
                i, j, state.triangles, state.octree, rng);
            if (factor > EPSILON) ++state.nonZeroKij;
            // This multiplication and clamp are invariant across timesteps.
            state.kij[state.idx2d(localI, j)] = std::min(factor * state.areas[j], ONE);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    std::vector<Vec3> centers(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) centers[i] = state.triangles[i].center();
    int minDelay = std::numeric_limits<int>::max();
    int maxDelay = 0;

    for (size_t localI = 0; localI < state.localTriangles; ++localI) {
        size_t i = state.firstTriangle + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            int delay = static_cast<int>(std::ceil((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
            state.tau[state.idx2d(localI, j)] = static_cast<uint8_t>(delay);
            minDelay = std::min(minDelay, delay);
            maxDelay = std::max(maxDelay, delay);
            // computeKij skips these pairs before making any random draws.
            if (!(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f)) {
                ++state.eligiblePairs;
            }
        }
    }
    int localBounds[2] = {minDelay, -maxDelay};
    int bounds[2];
    MPI_Allreduce(localBounds, bounds, 2, MPI_INT, MPI_MIN, state.comm);
    // Every non-self pair has a positive delay in this icosphere. A batch of
    // minDelay timesteps consequently uses only previously exchanged data.
    if (bounds[0] <= 0 || -bounds[1] > std::numeric_limits<uint8_t>::max()) {
        if (state.rank == 0) fprintf(stderr, "Unsupported propagation delay\n");
        MPI_Abort(state.comm, 1);
    }
    state.maxDelay = static_cast<size_t>(-bounds[1]);
    state.exchangeBatch = std::min(static_cast<size_t>(bounds[0]),
                                   std::max(size_t{1}, state.numTimesteps));
    state.historyLength = std::max(size_t{1}, std::min(state.maxDelay + 1, state.numTimesteps));
    state.history.resize(state.historyLength * state.numTriangles, ZERO);
    state.exchangeBuffer.resize(state.exchangeBatch * state.numTriangles);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    std::vector<int> counts(state.numRanks), displacements(state.numRanks);
    std::array<const val_t*, 256> delayedRows{};

    for (size_t firstT = 0; firstT < state.numTimesteps; firstT += state.exchangeBatch) {
        size_t batch = std::min(state.exchangeBatch, state.numTimesteps - firstT);
        for (size_t t = firstT; t < firstT + batch; ++t) {
            for (size_t d = 1; d <= std::min(t, state.maxDelay); ++d) {
                delayedRows[d] = state.history.data() + ((t - d) % state.historyLength) * state.numTriangles;
            }
            for (size_t localI = 0; localI < state.localTriangles; ++localI) {
                size_t i = state.firstTriangle + localI;
                const val_t* weights = state.kij.data() + localI * state.numTriangles;
                const uint8_t* delays = state.tau.data() + localI * state.numTriangles;
                val_t sumB = ZERO;
                // Preserve the serial accumulation order for each receiver.
                for (size_t j = 0; j < state.numTriangles; ++j) {
                    if (i == j || delays[j] > t || weights[j] <= ZERO) continue;
                    val_t radJ = delayedRows[delays[j]][j];
                    if (radJ <= ZERO) continue;
                    // Retain the original clamp expression so the compiler
                    // preserves its multiplication/addition rounding behavior.
                    sumB += std::min(weights[j], ONE) * radJ;
                }
                val_t emission = (i == state.sourceIndex && t < state.numTimesteps / 2) ? ONE : ZERO;
                state.radB[state.idxTN(t, localI)] = state.reflectivity * sumB + emission;
            }
            if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
            }
        }

        for (int r = 0; r < state.numRanks; ++r) {
            counts[r] = static_cast<int>(batch * static_cast<size_t>(state.rowCounts[r]));
            displacements[r] = static_cast<int>(batch * static_cast<size_t>(state.rowDisplacements[r]));
        }
        MPI_Allgatherv(state.radB.data() + state.idxTN(firstT, 0), counts[state.rank], MPI_FLOAT,
                       state.exchangeBuffer.data(), counts.data(), displacements.data(), MPI_FLOAT,
                       state.comm);
        // Allgatherv packs each rank's time block together; restore global row
        // order in the delay ring without ever gathering the full T x N history.
        for (size_t b = 0; b < batch; ++b) {
            val_t* row = state.history.data() + ((firstT + b) % state.historyLength) * state.numTriangles;
            for (int r = 0; r < state.numRanks; ++r) {
                std::copy_n(state.exchangeBuffer.data() + displacements[r] + b * state.rowCounts[r],
                            state.rowCounts[r], row + state.rowDisplacements[r]);
            }
            state.sourceHistory[firstT + b] = row[state.sourceIndex];
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    std::vector<val_t> signal(state.numTimesteps);
    std::vector<val_t> products(state.numTimesteps);

    for (size_t i = 0; i < state.localTriangles; ++i) {
        // A contiguous time series avoids strided reads in the quadratic loop.
        for (size_t t = 0; t < state.numTimesteps; ++t) signal[t] = state.radB[state.idxTN(t, i)];
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = signal[tt];
                val_t pS = state.sourceHistory[tt - t];
                products[tt - t] = pS * pB;
            }
            // Vectorize products while retaining the serial rounded products
            // and left-to-right sum, including nearly equal correlation peaks.
            for (size_t tt = 0; tt < state.numTimesteps - t; ++tt) sum += products[tt];

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
    std::vector<val_t> globalDistances;
    if (state.rank == 0) globalDistances.resize(state.numTriangles);
    MPI_Gatherv(state.distances.data(), state.rowCounts[state.rank], MPI_FLOAT,
                globalDistances.data(), state.rowCounts.data(), state.rowDisplacements.data(),
                MPI_FLOAT, 0, state.comm);
    if (state.rank == 0) state.distances.swap(globalDistances);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    uint64_t localCounts[2] = {0, state.nonZeroKij};
    for (size_t i = 0; i < state.localTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                ++localCounts[0];
                break;
            }
        }
    }
    uint64_t counts[2] = {};
    MPI_Reduce(localCounts, counts, 2, MPI_UINT64_T, MPI_SUM, 0, state.comm);
    if (state.rank != 0) return true;
    printf("\nValidation:\n");

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

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    uint64_t receivedEnergy = counts[0];
    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    uint64_t nonZeroKij = counts[1];
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
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
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

int runBenchmark(int argc, char** argv, int worldRank) {
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (worldRank == 0) fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (worldRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    if (!initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                              static_cast<size_t>(sourceIdx), reflectivity)) return 0;

    if (state.rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(state.comm);
    double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state);

    double preSeconds = MPI_Wtime() - startPre;
    double maxPreSeconds = 0;
    MPI_Reduce(&preSeconds, &maxPreSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, state.comm);
    long preDuration = static_cast<long>(maxPreSeconds * 1000);

    if (state.rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(state.comm);
    double startSim = MPI_Wtime();

    runSimulation(state);

    double simSeconds = MPI_Wtime() - startSim;
    double maxSimSeconds = 0;
    MPI_Reduce(&simSeconds, &maxSimSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, state.comm);
    long simDuration = static_cast<long>(maxSimSeconds * 1000);

    if (state.rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(state.comm);
    double startDist = MPI_Wtime();

    computeDistances(state);

    double distSeconds = MPI_Wtime() - startDist;
    double maxDistSeconds = 0;
    MPI_Reduce(&distSeconds, &maxDistSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, state.comm);
    long distDuration = static_cast<long>(maxDistSeconds * 1000);

    uint64_t localMem = state.kij.size() * sizeof(val_t) + state.tau.size() * sizeof(uint8_t)
                      + (state.radB.size() + state.history.size() + state.sourceHistory.size()
                         + state.exchangeBuffer.size() + state.distances.size() + state.areas.size()) * sizeof(val_t);
    uint64_t maxMem = 0;
    MPI_Reduce(&localMem, &maxMem, 1, MPI_UINT64_T, MPI_MAX, 0, state.comm);

    if (state.rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n) * n;
        double simOps = kijOps * t;
        double distOps = static_cast<double>(n) * t * t;

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        printf("  Memory usage per rank (max): %.2f MB\n", maxMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    // Validation
    int result = 0;
    if (validate) {
        if (!validateResults(state)) result = 1;
        MPI_Bcast(&result, 1, MPI_INT, 0, state.comm);
    }

    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "MPI rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
