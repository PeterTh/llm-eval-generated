/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed-memory implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light
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
#include <bit>
#include <climits>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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
    uint64_t state;

public:
    explicit RandomGenerator(uint64_t seed) : state(seed) {}

    val_t rand() {
        // SplitMix64 is inexpensive, has excellent equidistribution for this use,
        // and makes each triangle pair an independent deterministic work item.
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);
    }
};

uint64_t randomSeedForPair(size_t idxI, size_t idxJ, size_t numTriangles) {
    uint64_t pair = static_cast<uint64_t>(idxI) * static_cast<uint64_t>(numTriangles) +
                    static_cast<uint64_t>(idxJ);
    pair ^= pair >> 30;
    pair *= 0xbf58476d1ce4e5b9ULL;
    pair ^= pair >> 27;
    pair *= 0x94d049bb133111ebULL;
    pair ^= pair >> 31;
    return pair ^ 42ULL;
}

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
                  const Octree& octree) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    RandomGenerator rng(randomSeedForPair(idxI, idxJ, triangles.size()));
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
    size_t localBegin;
    size_t localEnd;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> transfer;    // Local rows of min(Kij * emitter area, 1)
    std::vector<uint8_t> tau;       // Local rows of the time-delay matrix
    std::vector<val_t> radB;        // Replicated reflected radiosity (T x N)
    std::vector<val_t> distances;   // Complete on rank zero after MPI_Gatherv

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    val_t reflectivity;
    uint64_t localNonZeroKij;

    size_t localRows() const { return localEnd - localBegin; }
    size_t idxLocal(size_t localI, size_t j) const { return localI * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

struct WorkDistribution {
    int rank;
    int size;
    size_t begin;
    size_t end;
    std::vector<int> counts;
    std::vector<int> displacements;

    WorkDistribution(size_t n, int mpiRank, int mpiSize)
        : rank(mpiRank), size(mpiSize), begin(0), end(0),
          counts(static_cast<size_t>(mpiSize)),
          displacements(static_cast<size_t>(mpiSize)) {
        for (int r = 0; r < size; ++r) {
            size_t first = n * static_cast<size_t>(r) / static_cast<size_t>(size);
            size_t last = n * static_cast<size_t>(r + 1) / static_cast<size_t>(size);
            counts[static_cast<size_t>(r)] = static_cast<int>(last - first);
            displacements[static_cast<size_t>(r)] = static_cast<int>(first);
        }
        begin = static_cast<size_t>(displacements[static_cast<size_t>(rank)]);
        end = begin + static_cast<size_t>(counts[static_cast<size_t>(rank)]);
    }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          const WorkDistribution& work) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.localNonZeroKij = 0;
    state.localBegin = work.begin;
    state.localEnd = work.end;

    if (work.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (work.rank == 0) printf("Building octree on each MPI rank...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Only the owned receiver rows of the quadratic matrices are allocated.
    state.transfer.resize(state.localRows() * state.numTriangles, ZERO);
    state.tau.resize(state.localRows() * state.numTriangles, 0);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    if (work.rank == 0) state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

int computeLocalMatrices(SimulationState& state, const WorkDistribution& work) {
    if (work.rank == 0) {
        printf("Computing distributed time delays (Tau) and form factors (Kij)...\n");
    }

    int localMinDelay = INT_MAX;
    for (size_t i = state.localBegin; i < state.localEnd; ++i) {
        size_t localI = i - state.localBegin;
        size_t rowOffset = state.idxLocal(localI, 0);
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            int delay = computeTau(state.triangles[i], state.triangles[j]);
            if (delay > static_cast<int>(std::numeric_limits<uint8_t>::max())) {
                fprintf(stderr, "Time delay exceeds compact storage range\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            state.tau[rowOffset + j] = static_cast<uint8_t>(delay);
            localMinDelay = std::min(localMinDelay, delay);
            val_t kij = computeKij(i, j, state.triangles, state.octree);
            if (kij > EPSILON) ++state.localNonZeroKij;
            state.transfer[rowOffset + j] = std::min(kij * state.areas[j], ONE);
        }
    }

    int globalMinDelay = INT_MAX;
    MPI_Allreduce(&localMinDelay, &globalMinDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalMinDelay == INT_MAX ? 1 : std::max(globalMinDelay, 1);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, const WorkDistribution& work,
                   int minimumDelay) {
    if (work.rank == 0) {
        printf("Running distributed wave propagation simulation...\n");
        printf("  Synchronizing in batches of %d timestep(s)\n", minimumDelay);
    }

    const size_t n = state.numTriangles;
    const size_t localRows = state.localRows();
    const size_t maxBatch = std::min(static_cast<size_t>(minimumDelay),
                                     state.numTimesteps);
    std::vector<val_t> sendBuffer(maxBatch * localRows);
    std::vector<val_t> receiveBuffer(maxBatch * n);
    std::vector<int> batchCounts(static_cast<size_t>(work.size));
    std::vector<int> batchDisplacements(static_cast<size_t>(work.size));

    for (size_t batchStart = 0; batchStart < state.numTimesteps;
         batchStart += maxBatch) {
        size_t batchSize = std::min(maxBatch, state.numTimesteps - batchStart);

        // No timestep in this batch can depend on another timestep in the same
        // batch: every non-self interaction is delayed by at least maxBatch.
        for (size_t q = 0; q < batchSize; ++q) {
            size_t t = batchStart + q;
            val_t* sendRow = sendBuffer.data() + q * localRows;
            for (size_t localI = 0; localI < localRows; ++localI) {
                size_t i = state.localBegin + localI;
                size_t rowOffset = state.idxLocal(localI, 0);
                const val_t* transferRow = state.transfer.data() + rowOffset;
                const uint8_t* tauRow = state.tau.data() + rowOffset;
                val_t sumB = ZERO;

                for (size_t j = 0; j < n; ++j) {
                    size_t tauij = static_cast<size_t>(tauRow[j]);

                    // Skip if wave hasn't yet propagated from j to i.
                    if (t < tauij) continue;

                    val_t coefficient = transferRow[j];
                    if (coefficient <= ZERO) continue;

                    // Get radiosity from the time when the source emitted it.
                    size_t srcTime = t - tauij;
                    val_t radJ = state.radB[state.idxTN(srcTime, j)];
                    if (radJ <= ZERO) continue;

                    sumB += coefficient * radJ;
                }

                // Update radiosity: reflection + emission.
                val_t emission = (i == state.sourceIndex &&
                                  t < state.numTimesteps / 2) ? ONE : ZERO;
                sendRow[localI] = state.reflectivity * sumB + emission;
            }
        }

        int displacement = 0;
        for (int r = 0; r < work.size; ++r) {
            size_t count = batchSize *
                           static_cast<size_t>(work.counts[static_cast<size_t>(r)]);
            if (count > static_cast<size_t>(INT_MAX)) {
                fprintf(stderr, "MPI batch count exceeds INT_MAX\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            batchCounts[static_cast<size_t>(r)] = static_cast<int>(count);
            batchDisplacements[static_cast<size_t>(r)] = displacement;
            displacement += static_cast<int>(count);
        }

        MPI_Allgatherv(sendBuffer.data(), static_cast<int>(batchSize * localRows),
                       MPI_FLOAT, receiveBuffer.data(), batchCounts.data(),
                       batchDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);

        // MPI_Allgatherv returns rank-major blocks; transpose just the small
        // communication buffer into the time-major radiosity history.
        for (int r = 0; r < work.size; ++r) {
            size_t rankRows = static_cast<size_t>(work.counts[static_cast<size_t>(r)]);
            size_t rankBegin = static_cast<size_t>(work.displacements[static_cast<size_t>(r)]);
            const val_t* rankBlock = receiveBuffer.data() +
                batchDisplacements[static_cast<size_t>(r)];
            for (size_t q = 0; q < batchSize; ++q) {
                std::copy_n(rankBlock + q * rankRows, rankRows,
                            state.radB.data() + state.idxTN(batchStart + q, rankBegin));
            }
        }

        if (work.rank == 0) {
            size_t completed = batchStart + batchSize;
            if (completed % 10 == 0 || completed == state.numTimesteps) {
                printf("  Timestep %zu/%zu\n", completed, state.numTimesteps);
            }
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, const WorkDistribution& work) {
    if (work.rank == 0) {
        printf("Computing distributed distances via cross-correlation...\n");
    }

    std::vector<val_t> localDistances(state.localRows(), ZERO);
    for (size_t localI = 0; localI < state.localRows(); ++localI) {
        size_t i = state.localBegin + localI;
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

        localDistances[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Gatherv(localDistances.data(), static_cast<int>(localDistances.size()), MPI_FLOAT,
                work.rank == 0 ? state.distances.data() : nullptr,
                work.counts.data(), work.displacements.data(), MPI_FLOAT, 0,
                MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, uint64_t nonZeroKij) {
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
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check the globally reduced Kij count (the matrix itself is distributed).
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
        uint32_t bits = std::bit_cast<uint32_t>(state.distances[i]);
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

size_t getTriangleCountForSubdivisions(int subdivisions) {
    size_t triangles = 20;
    for (int i = 0; i < subdivisions; ++i) triangles *= 4;
    return triangles;
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

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            showHelp = true;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseError = true;
            break;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (targetTriangles <= 0 || timesteps < 0 || !std::isfinite(reflectivity) ||
        reflectivity < ZERO || reflectivity > ONE) {
        if (rank == 0) {
            fprintf(stderr, "Invalid problem size, timestep count, or reflectivity\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    size_t actualTriangles = getTriangleCountForSubdivisions(subdivisions);
    if (actualTriangles > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Triangle count exceeds MPI count range\n");
        MPI_Finalize();
        return 1;
    }
    WorkDistribution work(actualTriangles, rank, worldSize);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n", worldSize);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, work);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double phaseStart = MPI_Wtime();
    int minimumDelay = computeLocalMatrices(state, work);
    MPI_Barrier(MPI_COMM_WORLD);
    double localDuration = MPI_Wtime() - phaseStart;
    double preSeconds = 0.0;
    MPI_Reduce(&localDuration, &preSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long preDuration = static_cast<long>(preSeconds * 1000.0);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    phaseStart = MPI_Wtime();
    runSimulation(state, work, minimumDelay);
    MPI_Barrier(MPI_COMM_WORLD);
    localDuration = MPI_Wtime() - phaseStart;
    double simSeconds = 0.0;
    MPI_Reduce(&localDuration, &simSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long simDuration = static_cast<long>(simSeconds * 1000.0);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    phaseStart = MPI_Wtime();
    computeDistances(state, work);
    MPI_Barrier(MPI_COMM_WORLD);
    localDuration = MPI_Wtime() - phaseStart;
    double distSeconds = 0.0;
    MPI_Reduce(&localDuration, &distSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long distDuration = static_cast<long>(distSeconds * 1000.0);

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    if (rank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);

        double kijOps = static_cast<double>(n) * static_cast<double>(n);
        double simOps = kijOps * static_cast<double>(t);
        double distOps = static_cast<double>(n) * static_cast<double>(t) *
                         static_cast<double>(t);

        printf("\nPerformance:\n");
        printf("  MPI ranks: %d\n", worldSize);
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        size_t maxRows = (n + static_cast<size_t>(worldSize) - 1) /
                         static_cast<size_t>(worldSize);
        size_t memMatrices = maxRows * n * (sizeof(val_t) + sizeof(uint8_t));
        size_t memRad = t * n * sizeof(val_t);
        size_t maxBatch = std::min(static_cast<size_t>(minimumDelay), t);
        size_t memCommunication = maxBatch * (n + maxRows) * sizeof(val_t);
        size_t totalMem = memMatrices + memRad + memCommunication;
        printf("  Peak array memory per rank: %.2f MB\n",
               totalMem / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    int validationPassed = 1;
    if (validate) {
        uint64_t globalNonZeroKij = 0;
        MPI_Reduce(&state.localNonZeroKij, &globalNonZeroKij, 1, MPI_UINT64_T,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            validationPassed = validateResults(state, globalNonZeroKij) ? 1 : 0;
        }
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
