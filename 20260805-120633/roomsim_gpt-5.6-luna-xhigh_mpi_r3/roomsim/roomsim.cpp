/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed-memory MPI implementation of a simplified room impulse
 * response simulation using radiosity-based wave propagation. It models how sound/light
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
#include <mpi.h>
#include <random>
#include <vector>

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
    void discard(uint64_t count) { rng.discard(count); }
};

// Generate a random point inside a triangle using barycentric coordinates
template <typename Rng>
Vec3 randomPointInTriangle(const Triangle& t, Rng& rng) {
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
template <typename Rng>
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  Rng& rng) {
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

    int mpiRank = 0;
    int mpiSize = 1;
    size_t localBegin = 0;
    size_t localCount = 0;
    int sourceOwner = 0;
    std::vector<int> mpiCounts;
    std::vector<int> mpiDisplacements;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Local area-weighted form-factor rows
    std::vector<int> tau;           // Local time-delay rows (local N x N)
    std::vector<val_t> radB;        // Local reflected radiosity (T x local N)
    std::vector<val_t> distances;   // Local distances from source

    // A ring of globally visible prior radiosity vectors.  Its depth is the
    // largest delay, so memory is O(N * maxTau), rather than O(T * N) per rank.
    std::vector<val_t> delayedRadiosity;
    std::vector<val_t> currentRadiosity;
    std::vector<val_t> sourceRadiosity;
    int maxTau = 0;
    long long localNonZeroFormFactors = 0;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t localIdx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t localIdxTN(size_t t, size_t i) const { return t * localCount + i; }

    void initializePartition() {
        mpiCounts.resize(static_cast<size_t>(mpiSize));
        mpiDisplacements.resize(static_cast<size_t>(mpiSize));

        const size_t base = numTriangles / static_cast<size_t>(mpiSize);
        const size_t remainder = numTriangles % static_cast<size_t>(mpiSize);
        size_t offset = 0;
        for (int rank = 0; rank < mpiSize; ++rank) {
            const size_t count = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                if (mpiRank == 0) {
                    std::fprintf(stderr, "Triangle count exceeds MPI int count limits\n");
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            mpiCounts[static_cast<size_t>(rank)] = static_cast<int>(count);
            mpiDisplacements[static_cast<size_t>(rank)] = static_cast<int>(offset);
            offset += count;
        }

        localBegin = static_cast<size_t>(mpiDisplacements[static_cast<size_t>(mpiRank)]);
        localCount = static_cast<size_t>(mpiCounts[static_cast<size_t>(mpiRank)]);
        sourceOwner = 0;
        for (int rank = 0; rank < mpiSize; ++rank) {
            const size_t begin = static_cast<size_t>(mpiDisplacements[static_cast<size_t>(rank)]);
            const size_t end = begin + static_cast<size_t>(mpiCounts[static_cast<size_t>(rank)]);
            if (sourceIndex >= begin && sourceIndex < end) {
                sourceOwner = rank;
                break;
            }
        }
    }
};

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

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    state.initializePartition();

    // Build octree for spatial acceleration
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.localCount * state.numTriangles, ZERO);
    state.tau.resize(state.localCount * state.numTriangles, 0);
    state.radB.resize(timesteps * state.localCount, ZERO);
    state.distances.resize(state.localCount, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    // Preserve the original single stream independent of the number of MPI
    // ranks.  Each rank owns a contiguous row range, so it can jump to the
    // exact stream position for its first row.  A non-culling pair consumes
    // four random values per ray (two barycentric values for each endpoint).
    RandomGenerator rng(42);
    uint64_t localRandomValues = 0;
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i != j &&
                state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) {
                localRandomValues += static_cast<uint64_t>(NUM_RAYS) * 4ULL;
            }
        }
    }
    uint64_t randomValuesBeforeLocalRows = 0;
    MPI_Exscan(&localRandomValues, &randomValuesBeforeLocalRows, 1,
               MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (state.mpiRank == 0) randomValuesBeforeLocalRows = 0;
    rng.discard(randomValuesBeforeLocalRows);

    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.localIdx2d(localI, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
        if (state.mpiRank == 0 &&
            ((i + 1) % 100 == 0 || i + 1 == state.numTriangles)) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }

    // The area-weighted coefficient is invariant throughout propagation.  Do
    // this once so the hot timestep loop only performs the required multiply
    // by the source radiosity.
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            val_t& coefficient = state.kij[state.localIdx2d(localI, j)];
            if (coefficient > EPSILON) ++state.localNonZeroFormFactors;
            if (coefficient > ZERO) {
                coefficient = std::min(coefficient * state.areas[j], ONE);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    int localMaxTau = 0;
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            const int delay = computeTau(
                state.triangles[i], state.triangles[j]);
            state.tau[state.localIdx2d(localI, j)] = delay;
            localMaxTau = std::max(localMaxTau, delay);
        }
    }
    MPI_Allreduce(&localMaxTau, &state.maxTau, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    const size_t ringSlots = static_cast<size_t>(state.maxTau) + 1;
    state.delayedRadiosity.assign(ringSlots * state.numTriangles, ZERO);
    state.currentRadiosity.assign(state.numTriangles, ZERO);

    // A zero delay would introduce an intra-timestep dependency in the
    // original i-ordered loop.  Normal icosphere levels have strictly
    // positive delays, but retain exact semantics for any degenerate input by
    // using an ordered MPI path when one occurs.
    int localHasZeroDelay = 0;
    for (size_t localI = 0; localI < state.localCount && localHasZeroDelay == 0; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i != j && state.tau[state.localIdx2d(localI, j)] == 0) {
                localHasZeroDelay = 1;
                break;
            }
        }
    }
    int hasZeroDelay = 0;
    MPI_Allreduce(&localHasZeroDelay, &hasZeroDelay, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (hasZeroDelay != 0) {
            // This path is rarely used, but mirrors the original sequential
            // ordering exactly for zero-delay triangle pairs.
            std::fill(state.currentRadiosity.begin(), state.currentRadiosity.end(), ZERO);
            for (size_t i = 0; i < state.numTriangles; ++i) {
                const int owner = [&state, i]() {
                    for (int rank = 0; rank < state.mpiSize; ++rank) {
                        const size_t begin = static_cast<size_t>(state.mpiDisplacements[static_cast<size_t>(rank)]);
                        const size_t end = begin + static_cast<size_t>(state.mpiCounts[static_cast<size_t>(rank)]);
                        if (i >= begin && i < end) return rank;
                    }
                    return 0;
                }();

                val_t value = ZERO;
                if (state.mpiRank == owner) {
                    const size_t localI = i - state.localBegin;
                    val_t sumB = ZERO;
                    for (size_t j = 0; j < state.numTriangles; ++j) {
                        if (i == j) continue;
                        const int tauij = state.tau[state.localIdx2d(localI, j)];
                        if (static_cast<int>(t) < tauij) continue;
                        const val_t coefficient = state.kij[state.localIdx2d(localI, j)];
                        if (coefficient <= ZERO) continue;

                        const size_t srcTime = t - static_cast<size_t>(tauij);
                        const val_t radJ = tauij == 0
                            ? state.currentRadiosity[j]
                            : state.delayedRadiosity[(srcTime % ringSlots) * state.numTriangles + j];
                        if (radJ > ZERO) sumB += coefficient * radJ;
                    }
                    const val_t emission = (i == state.sourceIndex && t < state.numTimesteps / 2)
                        ? ONE : ZERO;
                    value = state.rho[i] * sumB + emission;
                    state.radB[state.localIdxTN(t, localI)] = value;
                }
                MPI_Bcast(&value, 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
                state.currentRadiosity[i] = value;
            }
            std::copy(state.currentRadiosity.begin(), state.currentRadiosity.end(),
                      state.delayedRadiosity.begin() + (t % ringSlots) * state.numTriangles);
        } else {
            // All local receiver rows are independent at a given timestep.
            // Delayed source vectors are already globally visible in the ring.
            for (size_t localI = 0; localI < state.localCount; ++localI) {
                const size_t i = state.localBegin + localI;
                val_t sumB = ZERO;
                const size_t kijOffset = state.localIdx2d(localI, 0);
                for (size_t j = 0; j < state.numTriangles; ++j) {
                    const int tauij = state.tau[kijOffset + j];
                    if (static_cast<int>(t) < tauij) continue;

                    const val_t coefficient = state.kij[kijOffset + j];
                    if (coefficient <= ZERO) continue;

                    const size_t srcTime = t - static_cast<size_t>(tauij);
                    const val_t radJ = state.delayedRadiosity[
                        (srcTime % ringSlots) * state.numTriangles + j];
                    if (radJ > ZERO) sumB += coefficient * radJ;
                }

                const val_t emission = (i == state.sourceIndex && t < state.numTimesteps / 2)
                    ? ONE : ZERO;
                state.radB[state.localIdxTN(t, localI)] = state.rho[i] * sumB + emission;
            }

            // Make this timestep's distributed output available as a source
            // vector on every rank before advancing to t + 1.
            const val_t* localRadiosity = state.localCount == 0
                ? nullptr : state.radB.data() + t * state.localCount;
            MPI_Allgatherv(
                localRadiosity,
                static_cast<int>(state.localCount), MPI_FLOAT,
                state.currentRadiosity.data(), state.mpiCounts.data(),
                state.mpiDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
            std::copy(state.currentRadiosity.begin(), state.currentRadiosity.end(),
                      state.delayedRadiosity.begin() + (t % ringSlots) * state.numTriangles);
        }

        if (state.mpiRank == 0 &&
            ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    state.sourceRadiosity.assign(state.numTimesteps, ZERO);
    if (state.sourceIndex >= state.localBegin &&
        state.sourceIndex < state.localBegin + state.localCount) {
        const size_t localSource = state.sourceIndex - state.localBegin;
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            state.sourceRadiosity[t] = state.radB[state.localIdxTN(t, localSource)];
        }
    }
    MPI_Bcast(state.sourceRadiosity.data(), static_cast<int>(state.numTimesteps),
              MPI_FLOAT, state.sourceOwner, MPI_COMM_WORLD);

    for (size_t localI = 0; localI < state.localCount; ++localI) {
        val_t maxCorr = ZERO;
        int bestT = 0;
        const size_t iOffset = localI;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sumB = ZERO;
            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                const val_t pB = state.radB[state.localIdxTN(tt, iOffset)];
                const val_t pS = state.sourceRadiosity[tt - t];
                sumB += pS * pB;
            }

            if (sumB > maxCorr) {
                maxCorr = sumB;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    const int root = 0;

    // Check that distances are non-negative
    int localAllNonNegative = 1;
    int localFinite = 1;
    val_t localMinDist = std::numeric_limits<val_t>::max();
    val_t localMaxDist = std::numeric_limits<val_t>::lowest();
    val_t localSumDist = ZERO;
    long long localNonZeroCount = 0;

    for (const val_t d : state.distances) {
        if (d < 0) {
            localAllNonNegative = 0;
        }
        if (!std::isfinite(d)) {
            localFinite = 0;
        }
        localMinDist = std::min(localMinDist, d);
        localMaxDist = std::max(localMaxDist, d);
        localSumDist += d;
        if (d > EPSILON) ++localNonZeroCount;
    }

    int allNonNegative = 0;
    int allFinite = 0;
    val_t minDist = ZERO;
    val_t maxDist = ZERO;
    val_t sumDist = ZERO;
    long long nonZeroCount = 0;
    MPI_Allreduce(&localAllNonNegative, &allNonNegative, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMinDist, &minDist, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMaxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumDist, &sumDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNonZeroCount, &nonZeroCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    int localReceivedEnergy = 0;
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.localIdxTN(t, localI)] > EPSILON) {
                ++localReceivedEnergy;
                break;
            }
        }
    }
    int receivedEnergy = 0;
    MPI_Allreduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    long long localNonZeroKij = state.localNonZeroFormFactors;
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    val_t sourceDistance = ZERO;
    if (state.sourceIndex >= state.localBegin &&
        state.sourceIndex < state.localBegin + state.localCount) {
        sourceDistance = state.distances[state.sourceIndex - state.localBegin];
    }
    MPI_Bcast(&sourceDistance, 1, MPI_FLOAT, state.sourceOwner, MPI_COMM_WORLD);

    bool valid = allNonNegative != 0 && allFinite != 0 && receivedEnergy != 0 && nonZeroKij != 0;
    if (state.mpiRank == root) {
        printf("\nValidation:\n");
        if (allNonNegative == 0) printf("  ERROR: Negative distance detected\n");
        if (allFinite == 0) printf("  ERROR: Non-finite distance detected\n");
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %lld/%zu\n", nonZeroCount, state.numTriangles);

        // Check source distance is zero or very small
        if (sourceDistance > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", sourceDistance);
        }

        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

        const long long totalPairs = static_cast<long long>(state.numTriangles) *
                                     static_cast<long long>(state.numTriangles);
        printf("  Non-zero form factors: %lld/%lld (%.2f%%)\n",
               nonZeroKij, totalPairs,
               100.0f * static_cast<double>(nonZeroKij) / static_cast<double>(totalPairs));

        if (receivedEnergy == 0) {
            printf("  ERROR: No triangles received energy - simulation failed\n");
        }
        if (nonZeroKij == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
        }
        if (valid) printf("  Validation: PASSED\n");
    }

    return valid;
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

void gatherDistances(const SimulationState& state, std::vector<val_t>& globalDistances) {
    if (state.mpiRank == 0) globalDistances.resize(state.numTriangles);
    const val_t* localDistances = state.localCount == 0 ? nullptr : state.distances.data();
    MPI_Gatherv(
        localDistances, static_cast<int>(state.localCount), MPI_FLOAT,
        state.mpiRank == 0 ? globalDistances.data() : nullptr,
        state.mpiCounts.data(), state.mpiDisplacements.data(), MPI_FLOAT,
        0, MPI_COMM_WORLD);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpiRank = 0;
    int mpiSize = 1;
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n", mpiSize);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    double localPreDuration = std::chrono::duration<double, std::milli>(endPre - startPre).count();
    double preDurationMax = 0.0;
    MPI_Reduce(&localPreDuration, &preDurationMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long preDuration = mpiRank == 0 ? static_cast<long>(preDurationMax) : 0;

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    double localSimDuration = std::chrono::duration<double, std::milli>(endSim - startSim).count();
    double simDurationMax = 0.0;
    MPI_Reduce(&localSimDuration, &simDurationMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long simDuration = mpiRank == 0 ? static_cast<long>(simDurationMax) : 0;

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    double localDistDuration = std::chrono::duration<double, std::milli>(endDist - startDist).count();
    double distDurationMax = 0.0;
    MPI_Reduce(&localDistDuration, &distDurationMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long distDuration = mpiRank == 0 ? static_cast<long>(distDurationMax) : 0;

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    std::vector<val_t> globalDistances;
    gatherDistances(state, globalDistances);

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    if (mpiRank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (mpiRank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    }

    // Memory usage
    size_t memKij = state.localCount * n * sizeof(val_t);
    size_t memTau = state.localCount * n * sizeof(int);
    size_t memRad = t * state.localCount * sizeof(val_t);
    size_t memRing = state.delayedRadiosity.size() * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad + memRing;
    if (mpiRank == 0) {
        printf("  Per-rank memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
    }

    // Hash
    uint64_t hash = 0;
    if (mpiRank == 0) hash = computeHash(globalDistances);
    MPI_Bcast(&hash, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (mpiRank == 0) {
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        if (mpiRank == 0) {
            // Convert distances to double for output
            std::vector<double> distData(globalDistances.begin(), globalDistances.end());
            print_results(distData, "Distances");
        }
    }

    // Validation
    bool valid = true;
    if (validate) {
        valid = validateResults(state);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
