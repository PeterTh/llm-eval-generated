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
#include <vector>
#include <climits>
#include <cfloat>
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
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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
    void setState(const std::mt19937& state) { rng = state; }
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    int rank = 0, ranks = 1;
    size_t rowBegin = 0, rowEnd = 0;

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

    if (state.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    if (state.rank == 0) {
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.tau.resize(state.numTriangles * state.numTriangles, 0);
    }
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

// Flat octree representation for CUDA ray traversal. Child and leaf order match
// Octree::applyToTris, so visibility uses the same geometric tests.
struct DeviceNode {
    Vec3 center, halfExtent;
    int children[8];
    int begin, count;
};
struct RaySample { Vec3 from, to; };

static void cudaCheck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr)

static int flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes,
                         std::vector<int>& indices) {
    const int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    DeviceNode node{};
    node.center = tree.center;
    node.halfExtent = tree.halfExtent;
    node.begin = static_cast<int>(indices.size());
    node.count = static_cast<int>(tree.triangleIndices.size());
    for (size_t index : tree.triangleIndices) indices.push_back(static_cast<int>(index));
    for (int c = 0; c < 8; ++c)
        node.children[c] = tree.children[c] ? flattenOctree(*tree.children[c], nodes, indices) : -1;
    nodes[id] = node;
    return id;
}

__device__ static inline Vec3 dvsub(Vec3 a, Vec3 b) {
    return Vec3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ static inline float ddot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ static inline Vec3 dcross(Vec3 a, Vec3 b) {
    return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ static inline float dnorm(Vec3 a) { return sqrtf(ddot(a,a)); }
__device__ static inline bool boxHit(Vec3 p1, Vec3 p2, const DeviceNode& node) {
    Vec3 d = Vec3((p2.x-p1.x)*0.5f, (p2.y-p1.y)*0.5f, (p2.z-p1.z)*0.5f);
    Vec3 c = Vec3(p1.x+d.x-node.center.x, p1.y+d.y-node.center.y, p1.z+d.z-node.center.z);
    Vec3 ad = Vec3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    Vec3 h = node.halfExtent;
    if (fabsf(c.x)>h.x+ad.x || fabsf(c.y)>h.y+ad.y || fabsf(c.z)>h.z+ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y)>h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z)>h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x)>h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}
__device__ static inline float rayTriangle(Vec3 orig, Vec3 dir, const Triangle& tri) {
    Vec3 e1=dvsub(tri.b,tri.a), e2=dvsub(tri.c,tri.a);
    Vec3 pvec=dcross(dir,e2);
    float det=ddot(e1,pvec);
    if (fabsf(det)<EPSILON) return FLT_MAX;
    float invDet=1.0f/det;
    Vec3 tvec=dvsub(orig,tri.a);
    float u=ddot(tvec,pvec)*invDet;
    if (u<0.0f || u>1.0f) return FLT_MAX;
    Vec3 qvec=dcross(tvec,e1);
    float v=ddot(dir,qvec)*invDet;
    if (v<0.0f || u+v>1.0f) return FLT_MAX;
    return ddot(e2,qvec)*invDet;
}
__device__ static bool rayBlocked(Vec3 from, Vec3 to, const Triangle* tris,
                                   const DeviceNode* nodes, const int* indices,
                                   int src, int dst) {
    Vec3 delta=dvsub(to,from);
    float len=dnorm(delta);
    if (len<EPSILON) return true;
    Vec3 dir=Vec3(delta.x/len,delta.y/len,delta.z/len);
    // Depth-first traversal, one node per level on the stack.
    int stack[64], top=0;
    stack[top++]=0;
    while (top) {
        int id=stack[--top];
        const DeviceNode& node=nodes[id];
        if (node.count) {
            for (int k=0;k<node.count;++k) {
                int tri=indices[node.begin+k];
                if (tri==src || tri==dst) continue;
                float dist=rayTriangle(from,dir,tris[tri]);
                if (dist>EPSILON && dist<len-EPSILON) return true;
            }
        } else {
            for (int c=7;c>=0;--c) {
                int child=node.children[c];
                if (child>=0 && boxHit(from,to,nodes[child])) {
                    if (top>=64) return true; // Conservative for impossible pathological depth.
                    stack[top++]=child;
                }
            }
        }
    }
    return false;
}
__global__ static void formFactorKernel(const Triangle* tris, const DeviceNode* nodes,
                                         const int* indices, const RaySample* samples,
                                         float* kij, int* tau, int n, int rowBegin,
                                         int rows) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=rows*n) return;
    int i=rowBegin+q/n, j=q%n;
    int out=(i-rowBegin)*n+j;
    if (i==j) { kij[out]=0.0f; tau[out]=0; return; }
    const Triangle& a=tris[i];
    const Triangle& b=tris[j];
    Vec3 ca=Vec3((a.a.x+a.b.x+a.c.x)/3.0f,(a.a.y+a.b.y+a.c.y)/3.0f,(a.a.z+a.b.z+a.c.z)/3.0f);
    Vec3 cb=Vec3((b.a.x+b.b.x+b.c.x)/3.0f,(b.a.y+b.b.y+b.c.y)/3.0f,(b.a.z+b.b.z+b.c.z)/3.0f);
    tau[out]=static_cast<int>(ceilf(dnorm(dvsub(ca,cb))*INV_WAVE_SPEED));
    if (ddot(a._normal,b._normal)>0.99f) { kij[out]=0.0f; return; }
    float value=0.0f;
    for (int r=0;r<NUM_RAYS;++r) {
        RaySample sample=samples[q*NUM_RAYS+r];
        if (rayBlocked(sample.from,sample.to,tris,nodes,indices,i,j)) continue;
        Vec3 v=dvsub(sample.to,sample.from);
        float distSqr=ddot(v,v);
        if (distSqr<EPSILON) continue;
        float len=sqrtf(distSqr);
        float ci=fmaxf(0.0f,ddot(v,a._normal)/len);
        float cj=fmaxf(0.0f,-ddot(v,b._normal)/len);
        if (ci<=0.0f || cj<=0.0f) continue;
        value+=(ci*cj)/(PI*distSqr);
    }
    kij[out]=value*INV_NUM_RAYS;
}
__global__ static void propagationKernel(const float* kij, const int* tau,
                                           const float* area, const float* rad,
                                           float* out, int n, int t, int begin,
                                           int rows, int source, int timeOff,
                                           float reflectivity) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=rows) return;
    int i=begin+q;
    float sum=0.0f;
    for (int j=0;j<n;++j) {
        if (i==j) continue;
        int delay=tau[q*n+j];
        if (t<delay) continue;
        float k=kij[q*n+j];
        if (k<=0.0f) continue;
        float b=rad[(t-delay)*n+j];
        if (b<=0.0f) continue;
        sum+=fminf(k*area[j],1.0f)*b;
    }
    out[q]=reflectivity*sum+((i==source && t<timeOff)?1.0f:0.0f);
}
__global__ static void distanceKernel(const float* rad, float* distances,
                                       int n, int timesteps, int begin,
                                       int rows, int source) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=rows) return;
    int i=begin+q, best=0;
    float maxCorr=0.0f;
    for (int t=0;t<timesteps;++t) {
        float sum=0.0f;
        for (int tt=t;tt<timesteps;++tt)
            sum+=rad[(tt-t)*n+source]*rad[tt*n+i];
        if (sum>maxCorr) {maxCorr=sum; best=t;}
    }
    distances[q]=WAVE_SPEED*static_cast<float>(best);
}

struct GpuData {
    Triangle* triangles=nullptr;
    DeviceNode* nodes=nullptr;
    int* indices=nullptr;
    RaySample* samples=nullptr;
    float *kij=nullptr, *area=nullptr, *rad=nullptr, *out=nullptr, *distances=nullptr;
    int* tau=nullptr;
    size_t sampleCapacity=0;
    std::vector<float> localKij;
    std::vector<int> localTau;
    std::vector<float> localOut;
    std::vector<float> localDistances;
    ~GpuData() {
        cudaFree(triangles); cudaFree(nodes); cudaFree(indices); cudaFree(samples);
        cudaFree(kij); cudaFree(area); cudaFree(rad); cudaFree(out);
        cudaFree(distances); cudaFree(tau);
    }
};

static void prepareGpu(SimulationState& state, GpuData& gpu) {
    int localRank=0, deviceCount=0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,state.rank,MPI_INFO_NULL,&localComm);
    MPI_Comm_rank(localComm,&localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { fprintf(stderr,"CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(localRank%deviceCount));
    size_t n=state.numTriangles, rows=state.rowEnd-state.rowBegin;
    std::vector<DeviceNode> nodes;
    std::vector<int> indices;
    flattenOctree(state.octree,nodes,indices);
    CUDA_CHECK(cudaMalloc(&gpu.triangles,n*sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&gpu.nodes,nodes.size()*sizeof(DeviceNode)));
    CUDA_CHECK(cudaMalloc(&gpu.indices,indices.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu.triangles,state.triangles.data(),n*sizeof(Triangle),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.nodes,nodes.data(),nodes.size()*sizeof(DeviceNode),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.indices,indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.kij,rows*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.tau,rows*n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.area,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.rad,n*state.numTimesteps*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.out,rows*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.distances,rows*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.area,state.areas.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    gpu.localKij.resize(rows*n);
    gpu.localTau.resize(rows*n);
    gpu.localOut.resize(rows);
    gpu.localDistances.resize(rows);
}

static std::vector<std::mt19937> rowRandomStates(const SimulationState& state) {
    size_t n=state.numTriangles;
    std::vector<std::mt19937> states(n);
    std::mt19937 rng(42);
    // Every non-culled pair consumes exactly four uniform floats per ray.
    for (size_t i=0;i<n;++i) {
        states[i]=rng;
        size_t active=0;
        for (size_t j=0;j<n;++j)
            if (i!=j && state.triangles[i].normal().dot(state.triangles[j].normal())<=0.99f)
                ++active;
        rng.discard(active*NUM_RAYS*4);
    }
    return states;
}

static void computeMatrices(SimulationState& state, GpuData& gpu) {
    if (state.rank==0) printf("Computing time delays and form factors on CUDA...\n");
    size_t n=state.numTriangles, rows=state.rowEnd-state.rowBegin;
    auto seeds=rowRandomStates(state);
    constexpr size_t maxPairs=32768;
    size_t batchRows=std::max<size_t>(1,maxPairs/n);
    gpu.sampleCapacity=std::min(rows,batchRows)*n*NUM_RAYS;
    if (gpu.sampleCapacity) CUDA_CHECK(cudaMalloc(&gpu.samples,gpu.sampleCapacity*sizeof(RaySample)));
    std::vector<RaySample> samples(gpu.sampleCapacity);
    for (size_t begin=state.rowBegin;begin<state.rowEnd;begin+=batchRows) {
        size_t count=std::min(batchRows,state.rowEnd-begin);
        #pragma omp parallel for schedule(static)
        for (long long row=0;row<static_cast<long long>(count);++row) {
            size_t i=begin+row;
            RandomGenerator generator;
            generator.setState(seeds[i]);
            for (size_t j=0;j<n;++j) {
                if (i==j || state.triangles[i].normal().dot(state.triangles[j].normal())>0.99f) continue;
                for (int r=0;r<NUM_RAYS;++r) {
                    RaySample& s=samples[(row*n+j)*NUM_RAYS+r];
                    s.from=randomPointInTriangle(state.triangles[i],generator);
                    s.to=randomPointInTriangle(state.triangles[j],generator);
                }
            }
        }
        size_t pairCount=count*n;
        CUDA_CHECK(cudaMemcpy(gpu.samples,samples.data(),pairCount*NUM_RAYS*sizeof(RaySample),cudaMemcpyHostToDevice));
        int blocks=static_cast<int>((pairCount+127)/128);
        formFactorKernel<<<blocks,128>>>(gpu.triangles,gpu.nodes,gpu.indices,gpu.samples,
            gpu.kij+(begin-state.rowBegin)*n,gpu.tau+(begin-state.rowBegin)*n,
            static_cast<int>(n),static_cast<int>(begin),static_cast<int>(count));
        CUDA_CHECK(cudaGetLastError());
    }
    if (rows) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu.localKij.data(),gpu.kij,rows*n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu.localTau.data(),gpu.tau,rows*n*sizeof(int),cudaMemcpyDeviceToHost));
    }
    std::vector<int> counts(state.ranks),displs(state.ranks);
    for (int r=0;r<state.ranks;++r) {
        size_t begin=n*static_cast<size_t>(r)/state.ranks;
        size_t end=n*static_cast<size_t>(r+1)/state.ranks;
        counts[r]=static_cast<int>((end-begin)*n);
        displs[r]=static_cast<int>(begin*n);
    }
    MPI_Gatherv(gpu.localKij.data(),static_cast<int>(rows*n),MPI_FLOAT,
                state.kij.data(),counts.data(),displs.data(),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Gatherv(gpu.localTau.data(),static_cast<int>(rows*n),MPI_INT,
                state.tau.data(),counts.data(),displs.data(),MPI_INT,0,MPI_COMM_WORLD);
}

static void runSimulation(SimulationState& state, GpuData& gpu) {
    if (state.rank==0) printf("Running wave propagation simulation on CUDA...\n");
    size_t n=state.numTriangles, rows=state.rowEnd-state.rowBegin;
    std::vector<int> counts(state.ranks),displs(state.ranks);
    for (int r=0;r<state.ranks;++r) {
        size_t begin=n*static_cast<size_t>(r)/state.ranks;
        size_t end=n*static_cast<size_t>(r+1)/state.ranks;
        counts[r]=static_cast<int>(end-begin);
        displs[r]=static_cast<int>(begin);
    }
    for (size_t t=0;t<state.numTimesteps;++t) {
        if (rows) {
            propagationKernel<<<static_cast<int>((rows+127)/128),128>>>(gpu.kij,gpu.tau,gpu.area,
                gpu.rad,gpu.out,static_cast<int>(n),static_cast<int>(t),
                static_cast<int>(state.rowBegin),static_cast<int>(rows),
                static_cast<int>(state.sourceIndex),static_cast<int>(state.numTimesteps/2),
                state.rho[0]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(gpu.localOut.data(),gpu.out,rows*sizeof(float),cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(gpu.localOut.data(),static_cast<int>(rows),MPI_FLOAT,
            state.radB.data()+t*n,counts.data(),displs.data(),MPI_FLOAT,MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(gpu.rad+t*n,state.radB.data()+t*n,n*sizeof(float),cudaMemcpyHostToDevice));
        if (state.rank==0 && ((t+1)%10==0 || t+1==state.numTimesteps))
            printf("  Timestep %zu/%zu\n",t+1,state.numTimesteps);
    }
}

static void computeDistances(SimulationState& state, GpuData& gpu) {
    if (state.rank==0) printf("Computing distances via CUDA cross-correlation...\n");
    size_t n=state.numTriangles, rows=state.rowEnd-state.rowBegin;
    if (rows) {
        distanceKernel<<<static_cast<int>((rows+127)/128),128>>>(gpu.rad,gpu.distances,
            static_cast<int>(n),static_cast<int>(state.numTimesteps),
            static_cast<int>(state.rowBegin),static_cast<int>(rows),static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(gpu.localDistances.data(),gpu.distances,rows*sizeof(float),cudaMemcpyDeviceToHost));
    }
    std::vector<int> counts(state.ranks),displs(state.ranks);
    for (int r=0;r<state.ranks;++r) {
        size_t begin=n*static_cast<size_t>(r)/state.ranks;
        size_t end=n*static_cast<size_t>(r+1)/state.ranks;
        counts[r]=static_cast<int>(end-begin);
        displs[r]=static_cast<int>(begin);
    }
    MPI_Gatherv(gpu.localDistances.data(),static_cast<int>(rows),MPI_FLOAT,
                state.distances.data(),counts.data(),displs.data(),MPI_FLOAT,0,MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
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
    MPI_Init(&argc, &argv);
    int rank=0, ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
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
            if (rank==0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank==0) {
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
    state.rank=rank;
    state.ranks=ranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    state.rowBegin=state.numTriangles*static_cast<size_t>(rank)/ranks;
    state.rowEnd=state.numTriangles*static_cast<size_t>(rank+1)/ranks;
    GpuData gpu;
    prepareGpu(state,gpu);
    if (rank==0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeMatrices(state,gpu);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank==0) printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state,gpu);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank==0) printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state,gpu);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank==0) printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    long localTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank==0) {
    // Total time
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

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    }
    MPI_Finalize();
    return 0;
}
