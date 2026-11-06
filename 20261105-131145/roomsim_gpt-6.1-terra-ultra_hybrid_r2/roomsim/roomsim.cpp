/**
 * Room Response Simulation Benchmark
 * 
 * This is a parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <cfloat>
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
    std::vector<val_t> kij;         // Rank-local form factor rows
    std::vector<int> tau;           // Rank-local time delay rows
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const size_t localBegin = state.numTriangles * static_cast<size_t>(rank) / ranks;
    const size_t localEnd = state.numTriangles * static_cast<size_t>(rank + 1) / ranks;
    state.kij.resize((localEnd - localBegin) * state.numTriangles, ZERO);
    state.tau.resize((localEnd - localBegin) * state.numTriangles, 0);
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

// The pointer-based host octree is flattened once for traversal on the GPU.
struct DeviceOctreeNode {
    Vec3 center, halfExtent;
    int children[8];
    size_t first, count;
};

static int flattenOctree(const Octree& tree, std::vector<DeviceOctreeNode>& nodes,
                         std::vector<size_t>& indices) {
    int node = static_cast<int>(nodes.size());
    nodes.emplace_back();
    nodes[node].center = tree.center;
    nodes[node].halfExtent = tree.halfExtent;
    nodes[node].first = indices.size();
    nodes[node].count = tree.triangleIndices.size();
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (int i = 0; i < 8; ++i) {
        int child = tree.children[i] ? flattenOctree(*tree.children[i], nodes, indices) : -1;
        nodes[node].children[i] = child;
    }
    return node;
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

__device__ static Vec3 dadd(Vec3 a, Vec3 b) { Vec3 r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z; return r; }
__device__ static Vec3 dsub(Vec3 a, Vec3 b) { Vec3 r; r.x=a.x-b.x; r.y=a.y-b.y; r.z=a.z-b.z; return r; }
__device__ static Vec3 dscale(Vec3 a, float b) { Vec3 r; r.x=a.x*b; r.y=a.y*b; r.z=a.z*b; return r; }
__device__ static Vec3 ddivide(Vec3 a, float b) { Vec3 r; r.x=a.x/b; r.y=a.y/b; r.z=a.z/b; return r; }
__device__ static float ddot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ static Vec3 dcross(Vec3 a, Vec3 b) { Vec3 r; r.x=a.y*b.z-a.z*b.y; r.y=a.z*b.x-a.x*b.z; r.z=a.x*b.y-a.y*b.x; return r; }
__device__ static float dnorm(Vec3 a) { return sqrtf(ddot(a,a)); }

__device__ static bool deviceBoxHit(Vec3 from, Vec3 to, const DeviceOctreeNode& node) {
    Vec3 d=dscale(dsub(to,from),0.5f);
    Vec3 c=dsub(dadd(from,d),node.center);
    Vec3 ad; ad.x=fabsf(d.x); ad.y=fabsf(d.y); ad.z=fabsf(d.z);
    Vec3 h=node.halfExtent;
    if (fabsf(c.x)>h.x+ad.x || fabsf(c.y)>h.y+ad.y || fabsf(c.z)>h.z+ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y)>h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z)>h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x)>h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ static float deviceIntersection(Vec3 origin, Vec3 dir, const Triangle& tri) {
    Vec3 e1=dsub(tri.b,tri.a), e2=dsub(tri.c,tri.a);
    Vec3 p=dcross(dir,e2);
    float det=ddot(e1,p);
    if (fabsf(det)<EPSILON) return FLT_MAX;
    float inv=1.0f/det;
    Vec3 tv=dsub(origin,tri.a);
    float u=ddot(tv,p)*inv;
    if (u<0.0f || u>1.0f) return FLT_MAX;
    Vec3 q=dcross(tv,e1);
    float v=ddot(dir,q)*inv;
    if (v<0.0f || u+v>1.0f) return FLT_MAX;
    return ddot(e2,q)*inv;
}

__device__ static bool deviceBlocked(Vec3 from, Vec3 to, const Triangle* triangles,
                                     const DeviceOctreeNode* nodes, const size_t* indices,
                                     size_t src, size_t dst) {
    Vec3 dir=dsub(to,from);
    float len=dnorm(dir);
    if (len<EPSILON) return true;
    dir=ddivide(dir,len);
    // The tree stops when its bounding-box diagonal is below 0.5. With the
    // fixed radius-10 mesh its depth is at most seven; DFS needs <= 1+7*7 slots.
    int stack[64];
    int top=0;
    stack[top++]=0;
    while (top) {
        const DeviceOctreeNode& node=nodes[stack[--top]];
        if (node.count) {
            for (size_t k=0;k<node.count;++k) {
                size_t idx=indices[node.first+k];
                if (idx==src || idx==dst) continue;
                float dist=deviceIntersection(from,dir,triangles[idx]);
                if (dist>EPSILON && dist<len-EPSILON) return true;
            }
        } else {
            for (int k=7;k>=0;--k) {
                int child=node.children[k];
                if (child>=0 && deviceBoxHit(from,to,nodes[child])) stack[top++]=child;
            }
        }
    }
    return false;
}

__device__ static Vec3 devicePoint(const Triangle& tri, float u, float v) {
    if (u+v>1.0f) { u=1.0f-u; v=1.0f-v; }
    return dadd(dadd(tri.a,dscale(dsub(tri.b,tri.a),u)),dscale(dsub(tri.c,tri.a),v));
}

__global__ static void pairKernel(const Triangle* triangles, const DeviceOctreeNode* nodes,
                                  const size_t* indices, const float* samples,
                                  float* factors, int* delays, size_t n,
                                  size_t first, size_t pairCount) {
    size_t pair=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair>=pairCount) return;
    size_t i=first+pair/n, j=pair%n;
    if (i==j) { factors[pair]=0.0f; delays[pair]=0; return; }
    Triangle ti=triangles[i], tj=triangles[j];
    Vec3 ci=ddivide(dadd(dadd(ti.a,ti.b),ti.c),3.0f);
    Vec3 cj=ddivide(dadd(dadd(tj.a,tj.b),tj.c),3.0f);
    delays[pair]=static_cast<int>(ceilf(dnorm(dsub(ci,cj))*INV_WAVE_SPEED));
    if (ddot(ti._normal,tj._normal)>0.99f) { factors[pair]=0.0f; return; }
    float sum=0.0f;
    const float* values=samples+pair*(4*NUM_RAYS);
    for (int r=0;r<NUM_RAYS;++r) {
        Vec3 pi=devicePoint(ti,values[4*r],values[4*r+1]);
        Vec3 pj=devicePoint(tj,values[4*r+2],values[4*r+3]);
        if (deviceBlocked(pi,pj,triangles,nodes,indices,i,j)) continue;
        Vec3 v=dsub(pj,pi);
        float ds=ddot(v,v);
        if (ds<EPSILON) continue;
        float norm=dnorm(v);
        float c1=norm<=EPSILON ? 0.0f : fmaxf(0.0f,ddot(v,ti._normal)/norm);
        Vec3 neg; neg.x=-v.x; neg.y=-v.y; neg.z=-v.z;
        float c2=norm<=EPSILON ? 0.0f : fmaxf(0.0f,ddot(neg,tj._normal)/norm);
        if (c1<=0.0f || c2<=0.0f) continue;
        sum+=(c1*c2)/(PI*ds);
    }
    factors[pair]=sum*INV_NUM_RAYS;
}

// The original program consumes a single mt19937 stream only for pairs that
// pass the normal test. Generate the same stream in row-major order, in
// bounded batches, so results do not depend on rank count or GPU scheduling.
void cudaComputeFormFactors(SimulationState& state, size_t firstRow, size_t endRow) {
    const size_t n=state.numTriangles;
    if (firstRow==endRow) return;
    std::vector<DeviceOctreeNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree,nodes,indices);
    Triangle* dTriangles=nullptr;
    DeviceOctreeNode* dNodes=nullptr;
    size_t* dIndices=nullptr;
    float *dSamples=nullptr, *dFactors=nullptr;
    int* dDelays=nullptr;
    cudaCheck(cudaMalloc(&dTriangles,n*sizeof(Triangle)),"allocate triangles");
    cudaCheck(cudaMalloc(&dNodes,nodes.size()*sizeof(DeviceOctreeNode)),"allocate octree");
    cudaCheck(cudaMalloc(&dIndices,indices.size()*sizeof(size_t)),"allocate octree indices");
    cudaCheck(cudaMemcpy(dTriangles,state.triangles.data(),n*sizeof(Triangle),cudaMemcpyHostToDevice),"copy triangles");
    cudaCheck(cudaMemcpy(dNodes,nodes.data(),nodes.size()*sizeof(DeviceOctreeNode),cudaMemcpyHostToDevice),"copy octree");
    cudaCheck(cudaMemcpy(dIndices,indices.data(),indices.size()*sizeof(size_t),cudaMemcpyHostToDevice),"copy octree indices");
    const size_t batchRows=std::max<size_t>(1,std::min<size_t>(endRow-firstRow,(size_t(32)*1024*1024)/(n*4*NUM_RAYS*sizeof(float))));
    const size_t capacity=batchRows*n;
    std::vector<float> samples(capacity*4*NUM_RAYS);
    cudaCheck(cudaMalloc(&dSamples,samples.size()*sizeof(float)),"allocate random samples");
    cudaCheck(cudaMalloc(&dFactors,capacity*sizeof(float)),"allocate form factors");
    cudaCheck(cudaMalloc(&dDelays,capacity*sizeof(int)),"allocate delays");
    // Snapshot the stream at each row boundary. The snapshots let OpenMP
    // generate independent rows while preserving the serial draw sequence.
    std::vector<uint64_t> rowDraws(endRow, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i=0;i<endRow;++i) {
        uint64_t active=0;
        for (size_t j=0;j<n;++j)
            if (i!=j && state.triangles[i].normal().dot(state.triangles[j].normal())<=0.99f)
                ++active;
        rowDraws[i]=active*(4*NUM_RAYS);
    }
    RandomGenerator rng(42);
    uint64_t skipped=0;
    for (size_t i=0;i<firstRow;++i) skipped+=rowDraws[i];
    rng.discard(skipped);
    for (size_t row=firstRow;row<endRow;row+=batchRows) {
        size_t rows=std::min(batchRows,endRow-row), pairs=rows*n;
        std::vector<RandomGenerator> rowRngs;
        rowRngs.reserve(rows);
        for (size_t r=0;r<rows;++r) {
            rowRngs.push_back(rng);
            rng.discard(rowDraws[row+r]);
        }
        #pragma omp parallel for schedule(static)
        for (size_t r=0;r<rows;++r) {
            const size_t i=row+r;
            RandomGenerator localRng=rowRngs[r];
            for (size_t j=0;j<n;++j) {
                if (i==j || state.triangles[i].normal().dot(state.triangles[j].normal())>0.99f) continue;
                float* output=samples.data()+(r*n+j)*4*NUM_RAYS;
                for (int k=0;k<4*NUM_RAYS;++k) output[k]=localRng.rand();
            }
        }
        cudaCheck(cudaMemcpy(dSamples,samples.data(),pairs*4*NUM_RAYS*sizeof(float),cudaMemcpyHostToDevice),"copy random samples");
        pairKernel<<<static_cast<unsigned>((pairs+127)/128),128>>>(dTriangles,dNodes,dIndices,dSamples,dFactors,dDelays,n,row,pairs);
        cudaCheck(cudaGetLastError(),"launch form factor kernel");
        cudaCheck(cudaMemcpy(state.kij.data()+(row-firstRow)*n,dFactors,pairs*sizeof(float),cudaMemcpyDeviceToHost),"copy form factors");
        cudaCheck(cudaMemcpy(state.tau.data()+(row-firstRow)*n,dDelays,pairs*sizeof(int),cudaMemcpyDeviceToHost),"copy delays");
    }
    cudaFree(dSamples); cudaFree(dFactors); cudaFree(dDelays);
    cudaFree(dTriangles); cudaFree(dNodes); cudaFree(dIndices);
}

void cudaComputeTimeDelays(SimulationState&, size_t, size_t) {
    // Time delays are calculated alongside the form factors in pairKernel.
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Each MPI rank owns a contiguous range of receiving triangles.  Every rank
// keeps the time history, because each local receiver may depend on any source.
int mpiRank = 0, mpiSize = 1;
size_t firstRow = 0, endRow = 0;
std::vector<int> rowCounts, rowOffsets;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", mpiRank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void propagateRows(const float* kij, const int* tau, const float* areas,
                              const float* rho, const float* emission, float* radiosity,
                              size_t n, size_t row0, size_t rows, size_t t) {
    size_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= rows) return;
    size_t i = row0 + local;
    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t ij = local * n + j;
        int delay = tau[ij];
        if (t < static_cast<size_t>(delay)) continue;
        float factor = kij[ij];
        if (factor <= 0.0f) continue;
        float previous = radiosity[(t - static_cast<size_t>(delay)) * n + j];
        if (previous <= 0.0f) continue;
        sum += fminf(factor * areas[j], 1.0f) * previous;
    }
    radiosity[t * n + i] = rho[i] * sum + emission[t * n + i];
}

__global__ void correlateRows(const float* radiosity, float* distances,
                              size_t n, size_t steps, size_t source,
                              size_t row0, size_t rows) {
    size_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= rows) return;
    size_t i = row0 + local;
    float best = 0.0f;
    size_t bestT = 0;
    for (size_t t = 0; t < steps; ++t) {
        float sum = 0.0f;
        for (size_t tt = t; tt < steps; ++tt)
            sum += radiosity[(tt - t) * n + source] * radiosity[tt * n + i];
        if (sum > best) { best = sum; bestT = t; }
    }
    distances[local] = WAVE_SPEED * static_cast<float>(bestT);
}

void runSimulation(SimulationState& state) {
    if (mpiRank == 0) printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles, rows = endRow - firstRow;
    float *dK = nullptr, *dAreas = nullptr, *dRho = nullptr, *dE = nullptr, *dB = nullptr;
    int* dTau = nullptr;
    auto alloc = [](auto** ptr, size_t bytes) {
        if (bytes) checkCuda(cudaMalloc(reinterpret_cast<void**>(ptr), bytes), "cudaMalloc");
    };
    alloc(&dK, rows * n * sizeof(float));
    alloc(&dTau, rows * n * sizeof(int));
    alloc(&dAreas, n * sizeof(float));
    alloc(&dRho, n * sizeof(float));
    alloc(&dE, state.radE.size() * sizeof(float));
    alloc(&dB, state.radB.size() * sizeof(float));
    if (rows) {
        checkCuda(cudaMemcpy(dK, state.kij.data(), rows * n * sizeof(float), cudaMemcpyHostToDevice), "copy form factors");
        checkCuda(cudaMemcpy(dTau, state.tau.data(), rows * n * sizeof(int), cudaMemcpyHostToDevice), "copy delays");
    }
    checkCuda(cudaMemcpy(dAreas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy areas");
    checkCuda(cudaMemcpy(dRho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy reflectivity");
    if (!state.radE.empty()) {
        checkCuda(cudaMemcpy(dE, state.radE.data(), state.radE.size() * sizeof(float), cudaMemcpyHostToDevice), "copy emission");
        checkCuda(cudaMemcpy(dB, state.radB.data(), state.radB.size() * sizeof(float), cudaMemcpyHostToDevice), "copy radiosity");
    }
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (rows) {
            propagateRows<<<(rows + 127) / 128, 128>>>(dK, dTau, dAreas, dRho, dE, dB, n, firstRow, rows, t);
            checkCuda(cudaGetLastError(), "launch propagation");
            if (mpiSize > 1)
                checkCuda(cudaMemcpy(state.radB.data() + t * n + firstRow, dB + t * n + firstRow,
                                     rows * sizeof(float), cudaMemcpyDeviceToHost), "copy local radiosity");
        }
        if (mpiSize > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.radB.data() + t * n,
                           rowCounts.data(), rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(dB + t * n, state.radB.data() + t * n, n * sizeof(float), cudaMemcpyHostToDevice), "exchange radiosity");
        }
        if (mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps))
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
    }
    if (mpiSize == 1 && !state.radB.empty())
        checkCuda(cudaMemcpy(state.radB.data(), dB, state.radB.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy radiosity history");
    cudaFree(dK); cudaFree(dTau); cudaFree(dAreas); cudaFree(dRho); cudaFree(dE); cudaFree(dB);
}

void computeDistances(SimulationState& state) {
    if (mpiRank == 0) printf("Computing distances via cross-correlation...\n");
    const size_t rows = endRow - firstRow;
    float *dB = nullptr, *dDistances = nullptr;
    if (rows && !state.radB.empty()) {
        checkCuda(cudaMalloc(&dB, state.radB.size() * sizeof(float)), "allocate radiosity");
        checkCuda(cudaMalloc(&dDistances, rows * sizeof(float)), "allocate distances");
        checkCuda(cudaMemcpy(dB, state.radB.data(), state.radB.size() * sizeof(float), cudaMemcpyHostToDevice), "copy radiosity");
        correlateRows<<<(rows + 127) / 128, 128>>>(dB, dDistances, state.numTriangles,
            state.numTimesteps, state.sourceIndex, firstRow, rows);
        checkCuda(cudaGetLastError(), "launch correlation");
        checkCuda(cudaMemcpy(state.distances.data() + firstRow, dDistances, rows * sizeof(float), cudaMemcpyDeviceToHost), "copy distances");
        cudaFree(dB); cudaFree(dDistances);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), rowCounts.data(),
                   rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
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

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
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
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Keep benchmark output and result serialization on rank zero.
    if (mpiRank != 0 && !freopen("/dev/null", "w", stdout)) {
        fprintf(stderr, "Rank %d: failed to redirect output\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Assign one GPU per local MPI rank, respecting CUDA_VISIBLE_DEVICES.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    checkCuda(cudaGetDeviceCount(&deviceCount), "query CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "select CUDA device");

    rowCounts.resize(mpiSize);
    rowOffsets.resize(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        size_t begin = state.numTriangles * static_cast<size_t>(rank) / mpiSize;
        size_t end = state.numTriangles * static_cast<size_t>(rank + 1) / mpiSize;
        rowOffsets[rank] = static_cast<int>(begin);
        rowCounts[rank] = static_cast<int>(end - begin);
    }
    firstRow = static_cast<size_t>(rowOffsets[mpiRank]);
    endRow = firstRow + static_cast<size_t>(rowCounts[mpiRank]);

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    cudaComputeTimeDelays(state, firstRow, endRow);
    cudaComputeFormFactors(state, firstRow, endRow);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    MPI_Allreduce(MPI_IN_PLACE, &totalTime, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
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

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Count nonzero entries locally; the full matrix is never replicated.
    int valid = 1;
    if (validate) {
        uint64_t localNonZero = 0, globalNonZero = 0;
        #pragma omp parallel for reduction(+:localNonZero) schedule(static)
        for (size_t i = 0; i < state.kij.size(); ++i)
            localNonZero += state.kij[i] > EPSILON;
        MPI_Reduce(&localNonZero, &globalNonZero, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        if (mpiRank == 0) valid = validateResults(state, globalNonZero) ? 1 : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
