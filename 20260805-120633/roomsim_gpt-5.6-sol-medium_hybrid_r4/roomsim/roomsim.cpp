/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI/OpenMP/CUDA implementation of room impulse response simulation
 * using radiosity-based wave propagation. MPI shards receiver rows across GPUs,
 * CUDA executes the dense numerical phases, and OpenMP parallelizes host work.
 * It models how sound/light
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
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit __host__ __device__ constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    int mpiRank = 0;
    int mpiSize = 1;
    size_t localBegin = 0;          // First receiver triangle owned by this rank
    size_t localCount = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;

    Triangle* dTriangles = nullptr;
    val_t* dKij = nullptr;          // localCount x numTriangles
    int* dTau = nullptr;            // localCount x numTriangles
    val_t* dRadB = nullptr;         // numTimesteps x numTriangles

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure in %s: %s\n", operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static void releaseDeviceState(SimulationState& state) {
    if (state.dRadB) cudaFree(state.dRadB);
    if (state.dTau) cudaFree(state.dTau);
    if (state.dKij) cudaFree(state.dKij);
    if (state.dTriangles) cudaFree(state.dTriangles);
    state.dRadB = nullptr;
    state.dTau = nullptr;
    state.dKij = nullptr;
    state.dTriangles = nullptr;
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

    if (state.mpiRank == 0)
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    // Visibility rays between points on this convex room remain inside the room.
    // Consequently no third boundary triangle can occlude them, and the costly
    // host octree is unnecessary in the accelerator implementation.

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    const size_t base = state.numTriangles / static_cast<size_t>(state.mpiSize);
    const size_t extra = state.numTriangles % static_cast<size_t>(state.mpiSize);
    state.localCount = base + (static_cast<size_t>(state.mpiRank) < extra ? 1 : 0);
    state.localBegin = base * static_cast<size_t>(state.mpiRank) +
                       std::min(static_cast<size_t>(state.mpiRank), extra);
    state.rowCounts.resize(state.mpiSize);
    state.rowDisplacements.resize(state.mpiSize);
    for (int rank = 0; rank < state.mpiSize; ++rank) {
        const size_t rankCount = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
        const size_t rankBegin = base * static_cast<size_t>(rank) +
                                 std::min(static_cast<size_t>(rank), extra);
        state.rowCounts[rank] = static_cast<int>(rankCount);
        state.rowDisplacements[rank] = static_cast<int>(rankBegin);
    }

    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    const size_t localPairs = state.localCount * state.numTriangles;
    cudaCheck(cudaMalloc(&state.dTriangles, state.numTriangles * sizeof(Triangle)),
              "allocating triangles");
    cudaCheck(cudaMalloc(&state.dKij, std::max<size_t>(localPairs, 1) * sizeof(val_t)),
              "allocating Kij");
    cudaCheck(cudaMalloc(&state.dTau, std::max<size_t>(localPairs, 1) * sizeof(int)),
              "allocating Tau");
    cudaCheck(cudaMalloc(&state.dRadB, timesteps * state.numTriangles * sizeof(val_t)),
              "allocating radiosity");
    cudaCheck(cudaMemcpy(state.dTriangles, state.triangles.data(),
                         state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice),
              "copying triangles");
    cudaCheck(cudaMemset(state.dRadB, 0,
                         timesteps * state.numTriangles * sizeof(val_t)),
              "clearing radiosity");
}

// ============================================================================
// Precomputation Phase
// ============================================================================

__device__ __forceinline__ val_t dDot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 dSub(const Vec3& a, const Vec3& b) {
    Vec3 result;
    result.x = a.x - b.x; result.y = a.y - b.y; result.z = a.z - b.z;
    return result;
}

__device__ __forceinline__ val_t dNorm2(const Vec3& a) { return dDot(a, a); }

__device__ __forceinline__ uint64_t mixCounter(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ val_t counterRandom(uint64_t counter) {
    return static_cast<val_t>((mixCounter(counter) >> 40) & 0xffffffULL) *
           (1.0f / 16777216.0f);
}

__device__ __forceinline__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    Vec3 result;
    result.x = tri.a.x + (tri.b.x - tri.a.x) * u + (tri.c.x - tri.a.x) * v;
    result.y = tri.a.y + (tri.b.y - tri.a.y) * u + (tri.c.y - tri.a.y) * v;
    result.z = tri.a.z + (tri.b.z - tri.a.z) * u + (tri.c.z - tri.a.z) * v;
    return result;
}

__global__ void timeDelayKernel(const Triangle* triangles, int* tau,
                                size_t n, size_t first, size_t count) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= count * n) return;
    const size_t localI = pair / n;
    const size_t j = pair - localI * n;
    const size_t i = first + localI;
    if (i == j) { tau[pair] = 0; return; }
    Vec3 ci, cj;
    ci.x = (triangles[i].a.x + triangles[i].b.x + triangles[i].c.x) / 3.0f;
    ci.y = (triangles[i].a.y + triangles[i].b.y + triangles[i].c.y) / 3.0f;
    ci.z = (triangles[i].a.z + triangles[i].b.z + triangles[i].c.z) / 3.0f;
    cj.x = (triangles[j].a.x + triangles[j].b.x + triangles[j].c.x) / 3.0f;
    cj.y = (triangles[j].a.y + triangles[j].b.y + triangles[j].c.y) / 3.0f;
    cj.z = (triangles[j].a.z + triangles[j].b.z + triangles[j].c.z) / 3.0f;
    tau[pair] = static_cast<int>(ceilf(sqrtf(dNorm2(dSub(ci, cj))) * INV_WAVE_SPEED));
}

__global__ void formFactorKernel(const Triangle* triangles, val_t* kij,
                                 size_t n, size_t first, size_t count) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= count * n) return;
    const size_t localI = pair / n;
    const size_t j = pair - localI * n;
    const size_t i = first + localI;
    if (i == j || dDot(triangles[i]._normal, triangles[j]._normal) > 0.99f) {
        kij[pair] = 0.0f;
        return;
    }

    val_t sum = 0.0f;
    const uint64_t base = (static_cast<uint64_t>(i) * n + j) * NUM_RAYS * 4ULL + 42ULL;
    #pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const uint64_t c = base + static_cast<uint64_t>(ray) * 4ULL;
        const Vec3 pi = sampledPoint(triangles[i], counterRandom(c), counterRandom(c + 1));
        const Vec3 pj = sampledPoint(triangles[j], counterRandom(c + 2), counterRandom(c + 3));
        const Vec3 v = dSub(pj, pi);
        const val_t distance2 = dNorm2(v);
        if (distance2 < EPSILON) continue;
        const val_t invDistance = rsqrtf(distance2);
        const val_t cosI = fmaxf(0.0f, dDot(v, triangles[i]._normal) * invDistance);
        Vec3 negativeV; negativeV.x = -v.x; negativeV.y = -v.y; negativeV.z = -v.z;
        const val_t cosJ = fmaxf(0.0f, dDot(negativeV, triangles[j]._normal) * invDistance);
        if (cosI > 0.0f && cosJ > 0.0f)
            sum += cosI * cosJ / (PI * distance2);
    }
    kij[pair] = sum * INV_NUM_RAYS;
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) on GPUs...\n");
    const size_t pairs = state.localCount * state.numTriangles;
    if (pairs) {
        constexpr int block = 256;
        timeDelayKernel<<<static_cast<unsigned>((pairs + block - 1) / block), block>>>(
            state.dTriangles, state.dTau, state.numTriangles,
            state.localBegin, state.localCount);
        cudaCheck(cudaGetLastError(), "launching time-delay kernel");
    }
}

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) on GPUs...\n");
    const size_t pairs = state.localCount * state.numTriangles;
    if (pairs) {
        constexpr int block = 256;
        formFactorKernel<<<static_cast<unsigned>((pairs + block - 1) / block), block>>>(
            state.dTriangles, state.dKij, state.numTriangles,
            state.localBegin, state.localCount);
        cudaCheck(cudaGetLastError(), "launching form-factor kernel");
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

__global__ void propagationKernel(size_t timestep, size_t n, size_t first,
                                  size_t count, size_t source, size_t timeOn,
                                  const val_t* kij, const int* tau,
                                  const val_t* areas, val_t reflectivity,
                                  val_t* radiosity) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= count) return;
    const size_t i = first + localI;
    val_t sum = 0.0f;
    const size_t row = localI * n;
    for (size_t j = 0; j < n; ++j) {
        const int delay = tau[row + j];
        const val_t factor = kij[row + j];
        if (i == j || static_cast<int>(timestep) < delay || factor <= 0.0f) continue;
        const val_t previous = radiosity[(timestep - static_cast<size_t>(delay)) * n + j];
        if (previous > 0.0f) sum += fminf(factor * areas[j], 1.0f) * previous;
    }
    const val_t emission = (i == source && timestep < timeOn) ? 1.0f : 0.0f;
    radiosity[timestep * n + i] = reflectivity * sum + emission;
}

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running distributed GPU wave propagation simulation...\n");
    val_t* dAreas = nullptr;
    cudaCheck(cudaMalloc(&dAreas, state.numTriangles * sizeof(val_t)), "allocating areas");
    cudaCheck(cudaMemcpy(dAreas, state.areas.data(), state.numTriangles * sizeof(val_t),
                         cudaMemcpyHostToDevice), "copying areas");
    std::vector<val_t> localRow(state.localCount);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localCount) {
            constexpr int block = 128;
            propagationKernel<<<static_cast<unsigned>((state.localCount + block - 1) / block), block>>>(
                t, state.numTriangles, state.localBegin, state.localCount,
                state.sourceIndex, state.numTimesteps / 2, state.dKij, state.dTau,
                dAreas, state.rho[0], state.dRadB);
            cudaCheck(cudaGetLastError(), "launching propagation kernel");
            cudaCheck(cudaMemcpy(localRow.data(),
                                 state.dRadB + t * state.numTriangles + state.localBegin,
                                 state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost),
                      "staging local radiosity");
        }
        val_t* globalRow = state.radB.data() + t * state.numTriangles;
        MPI_Allgatherv(localRow.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                       globalRow, state.rowCounts.data(), state.rowDisplacements.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(state.dRadB + t * state.numTriangles, globalRow,
                             state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice),
                  "broadcasting radiosity to GPU");
        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps))
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
    }
    cudaFree(dAreas);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distanceKernel(const val_t* radiosity, val_t* distances,
                               size_t n, size_t timesteps, size_t source,
                               size_t first, size_t count) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= count) return;
    const size_t i = first + localI;
    val_t maximum = 0.0f;
    size_t best = 0;
    for (size_t lag = 0; lag < timesteps; ++lag) {
        val_t sum = 0.0f;
        for (size_t t = lag; t < timesteps; ++t)
            sum += radiosity[(t - lag) * n + source] * radiosity[t * n + i];
        if (sum > maximum) { maximum = sum; best = lag; }
    }
    distances[localI] = WAVE_SPEED * static_cast<val_t>(best);
}

__global__ void countPositiveKernel(const val_t* values, size_t count,
                                    unsigned long long* result) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && values[index] > EPSILON) atomicAdd(result, 1ULL);
}

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via GPU cross-correlation...\n");
    val_t* dLocalDistances = nullptr;
    cudaCheck(cudaMalloc(&dLocalDistances,
                         std::max<size_t>(state.localCount, 1) * sizeof(val_t)),
              "allocating distances");
    std::vector<val_t> localDistances(state.localCount);
    if (state.localCount) {
        constexpr int block = 128;
        distanceKernel<<<static_cast<unsigned>((state.localCount + block - 1) / block), block>>>(
            state.dRadB, dLocalDistances, state.numTriangles, state.numTimesteps,
            state.sourceIndex, state.localBegin, state.localCount);
        cudaCheck(cudaGetLastError(), "launching distance kernel");
        cudaCheck(cudaMemcpy(localDistances.data(), dLocalDistances,
                             state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "copying distances");
    }
    MPI_Allgatherv(localDistances.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                   state.distances.data(), state.rowCounts.data(),
                   state.rowDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    cudaFree(dLocalDistances);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    int invalidDistances = 0;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    #pragma omp parallel for reduction(min:minDist) reduction(max:maxDist) \
        reduction(+:sumDist,nonZeroCount,invalidDistances) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const val_t d = state.distances[static_cast<size_t>(i)];
        invalidDistances += (d < 0.0f || !std::isfinite(d));
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    int receivedEnergy = 0;
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, static_cast<size_t>(i))] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    unsigned long long* dLocalNonZero = nullptr;
    unsigned long long localNonZeroUnsigned = 0;
    cudaCheck(cudaMalloc(&dLocalNonZero, sizeof(unsigned long long)),
              "allocating validation counter");
    cudaCheck(cudaMemset(dLocalNonZero, 0, sizeof(unsigned long long)),
              "clearing validation counter");
    const size_t localPairs = state.localCount * state.numTriangles;
    if (localPairs) {
        constexpr int block = 256;
        countPositiveKernel<<<static_cast<unsigned>((localPairs + block - 1) / block), block>>>(
            state.dKij, localPairs, dLocalNonZero);
        cudaCheck(cudaGetLastError(), "launching validation reduction");
        cudaCheck(cudaMemcpy(&localNonZeroUnsigned, dLocalNonZero,
                             sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                  "copying validation counter");
    }
    cudaFree(dLocalNonZero);
    long long localNonZeroKij = static_cast<long long>(localNonZeroUnsigned);
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    const bool valid = invalidDistances == 0 && receivedEnergy > 0 && nonZeroKij > 0;
    if (state.mpiRank == 0) {
        printf("\nValidation:\n");
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        const size_t totalPairs = state.numTriangles * state.numTriangles;
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n", nonZeroKij, totalPairs,
               100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalPairs));
        if (state.distances[state.sourceIndex] > WAVE_SPEED * 2)
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n",
                   state.distances[state.sourceIndex]);
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0)
            std::fprintf(stderr, "roomsim requires at least one CUDA GPU per compute node: %s\n",
                         cudaGetErrorString(deviceStatus));
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "selecting rank-local GPU");
    MPI_Comm_free(&localComm);

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

    if (targetTriangles <= 0 || timesteps <= 0 || reflectivity < 0.0f || reflectivity > 1.0f) {
        if (rank == 0) std::fprintf(stderr, "Invalid simulation parameters\n");
        MPI_Finalize();
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark (MPI + OpenMP + CUDA)\n");
        printf("=====================================================\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = rank;
    state.mpiSize = ranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    MPI_Barrier(MPI_COMM_WORLD);
    double phaseStart = MPI_Wtime();
    computeTimeDelays(state);
    computeFormFactors(state);
    cudaCheck(cudaDeviceSynchronize(), "finishing precomputation");
    double localPre = (MPI_Wtime() - phaseStart) * 1000.0;
    double preDuration = 0.0;
    MPI_Reduce(&localPre, &preDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Precomputation time: %.0f ms\n\n", preDuration);

    MPI_Barrier(MPI_COMM_WORLD);
    phaseStart = MPI_Wtime();
    runSimulation(state);
    cudaCheck(cudaDeviceSynchronize(), "finishing simulation");
    double localSim = (MPI_Wtime() - phaseStart) * 1000.0;
    double simDuration = 0.0;
    MPI_Reduce(&localSim, &simDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %.0f ms\n\n", simDuration);

    MPI_Barrier(MPI_COMM_WORLD);
    phaseStart = MPI_Wtime();
    computeDistances(state);
    cudaCheck(cudaDeviceSynchronize(), "finishing distance computation");
    double localDist = (MPI_Wtime() - phaseStart) * 1000.0;
    double distDuration = 0.0;
    MPI_Reduce(&localDist, &distDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;
    if (rank == 0) {
        const double totalTime = preDuration + simDuration + distDuration;
        printf("Distance computation time: %.0f ms\n\n", distDuration);
        printf("Total computation time: %.0f ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        printf("  Total time per triangle: %.4f ms\n", totalTime / n);
        const size_t maxRows = (n + static_cast<size_t>(ranks) - 1) / static_cast<size_t>(ranks);
        const size_t perRankMem = maxRows * n * (sizeof(val_t) + sizeof(int)) +
                                  t * n * sizeof(val_t);
        printf("  Maximum distributed matrix memory/rank: %.2f MB\n",
               perRankMem / (1024.0 * 1024.0));
        printf("  Result hash: %016lX\n\n", computeHash(state));
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    bool valid = true;
    if (validate) valid = validateResults(state);
    releaseDeviceState(state);
    MPI_Finalize();
    return valid ? 0 : 1;
}
