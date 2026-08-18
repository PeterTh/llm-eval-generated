/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response simulation
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

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t cuda_status_ = (call);                                          \
    if (cuda_status_ != cudaSuccess) {                                          \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(cuda_status_));                         \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(cuda_status_));              \
    }                                                                           \
} while (false)

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
        triangles.resize(faces.size());
#pragma omp parallel for schedule(static)
        for (ptrdiff_t f = 0; f < static_cast<ptrdiff_t>(faces.size()); ++f) {
            const auto& face = faces[static_cast<size_t>(f)];
            // Reverse winding to make normals point inward
            triangles[static_cast<size_t>(f)] =
                Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
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
    size_t localRowBegin = 0;
    size_t localRows = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;

    Triangle* dTriangles = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dKij = nullptr;
    int* dTau = nullptr;
    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;

    size_t idx2d(size_t i, size_t j) const { return (i - localRowBegin) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Counter-based random numbers make every (triangle pair, ray) independent of
// launch shape, MPI rank count, and OpenMP scheduling.
__device__ __forceinline__ uint32_t mixBits(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__device__ __forceinline__ val_t sample01(uint64_t pair, int ray, int lane) {
    uint32_t x = static_cast<uint32_t>(pair) ^ static_cast<uint32_t>(pair >> 32);
    x ^= 0x9e3779b9u * static_cast<uint32_t>(4 * ray + lane + 1);
    return static_cast<val_t>(mixBits(x) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ Vec3 sampledPoint(const Triangle& tri, uint64_t pair,
                                              int ray, int lane) {
    val_t u = sample01(pair, ray, lane);
    val_t v = sample01(pair, ray, lane + 1);
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return {tri.a.x + (tri.b.x - tri.a.x) * u + (tri.c.x - tri.a.x) * v,
            tri.a.y + (tri.b.y - tri.a.y) * u + (tri.c.y - tri.a.y) * v,
            tri.a.z + (tri.b.z - tri.a.z) * u + (tri.c.z - tri.a.z) * v};
}

__global__ void precomputeKernel(const Triangle* __restrict__ triangles,
                                 val_t* __restrict__ kij, int* __restrict__ tau,
                                 size_t n, size_t firstRow, size_t rows) {
    size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = rows * n;
    if (q >= total) return;
    size_t localI = q / n;
    size_t i = firstRow + localI;
    size_t j = q - localI * n;
    if (i == j) { kij[q] = 0.0f; tau[q] = 0; return; }

    const Triangle& ti = triangles[i];
    const Triangle& tj = triangles[j];
    val_t cix = (ti.a.x + ti.b.x + ti.c.x) * (1.0f / 3.0f);
    val_t ciy = (ti.a.y + ti.b.y + ti.c.y) * (1.0f / 3.0f);
    val_t ciz = (ti.a.z + ti.b.z + ti.c.z) * (1.0f / 3.0f);
    val_t cjx = (tj.a.x + tj.b.x + tj.c.x) * (1.0f / 3.0f);
    val_t cjy = (tj.a.y + tj.b.y + tj.c.y) * (1.0f / 3.0f);
    val_t cjz = (tj.a.z + tj.b.z + tj.c.z) * (1.0f / 3.0f);
    val_t dx = cix - cjx, dy = ciy - cjy, dz = ciz - cjz;
    tau[q] = static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy + dz * dz) * INV_WAVE_SPEED));

    val_t normalDot = ti._normal.x * tj._normal.x +
                      ti._normal.y * tj._normal.y + ti._normal.z * tj._normal.z;
    if (normalDot > 0.99f) { kij[q] = 0.0f; return; }

    // The generated icosphere is convex.  A chord between two boundary facets
    // lies wholly inside it, so no third boundary facet can occlude the chord.
    // This removes irregular octree traversal without changing visibility.
    val_t sum = 0.0f;
    uint64_t pair = static_cast<uint64_t>(i) * n + j;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pi = sampledPoint(ti, pair, r, 0);
        Vec3 pj = sampledPoint(tj, pair, r, 2);
        val_t vx = pj.x - pi.x, vy = pj.y - pi.y, vz = pj.z - pi.z;
        val_t ds = vx * vx + vy * vy + vz * vz;
        if (ds < EPSILON) continue;
        val_t invLen = rsqrtf(ds);
        val_t ci = fmaxf(0.0f, (vx * ti._normal.x + vy * ti._normal.y + vz * ti._normal.z) * invLen);
        val_t cj = fmaxf(0.0f, (-vx * tj._normal.x - vy * tj._normal.y - vz * tj._normal.z) * invLen);
        sum += (ci * cj) / (PI * ds);
    }
    kij[q] = sum * INV_NUM_RAYS;
}

__global__ void propagationKernel(size_t t, size_t n, size_t firstRow, size_t rows,
                                  size_t source, size_t timeOn,
                                  const val_t* __restrict__ kij,
                                  const int* __restrict__ tau,
                                  const val_t* __restrict__ areas,
                                  const val_t* __restrict__ rho,
                                  val_t* __restrict__ radB) {
    size_t li = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= rows) return;
    size_t i = firstRow + li;
    val_t sum = 0.0f;
    size_t row = li * n;
    for (size_t j = 0; j < n; ++j) {
        int delay = tau[row + j];
        if (i == j || static_cast<int>(t) < delay) continue;
        val_t k = kij[row + j];
        if (k <= 0.0f) continue;
        val_t b = radB[(t - static_cast<size_t>(delay)) * n + j];
        if (b > 0.0f) sum += fminf(k * areas[j], 1.0f) * b;
    }
    radB[t * n + i] = rho[i] * sum + ((i == source && t < timeOn) ? 1.0f : 0.0f);
}

__global__ void distanceKernel(size_t n, size_t timesteps, size_t firstRow,
                               size_t rows, size_t source,
                               const val_t* __restrict__ radB,
                               val_t* __restrict__ distances) {
    size_t li = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= rows) return;
    size_t i = firstRow + li;
    val_t maxCorr = 0.0f;
    size_t bestT = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = 0.0f;
        for (size_t tt = t; tt < timesteps; ++tt)
            sum += radB[(tt - t) * n + source] * radB[tt * n + i];
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
    }
    distances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countPositiveKernel(const val_t* values, size_t count,
                                    unsigned long long* result) {
    size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int present = q < count && values[q] > EPSILON;
    // Reduce within a warp so only one atomic is issued per 32 elements.
    unsigned mask = __ballot_sync(0xffffffffu, present);
    if ((threadIdx.x & 31u) == 0 && mask != 0)
        atomicAdd(result, static_cast<unsigned long long>(__popc(mask)));
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

    size_t base = state.numTriangles / static_cast<size_t>(state.mpiSize);
    size_t extra = state.numTriangles % static_cast<size_t>(state.mpiSize);
    state.localRows = base + (static_cast<size_t>(state.mpiRank) < extra ? 1 : 0);
    state.localRowBegin = base * static_cast<size_t>(state.mpiRank) +
                          std::min(static_cast<size_t>(state.mpiRank), extra);
    state.rowCounts.resize(state.mpiSize);
    state.rowDispls.resize(state.mpiSize);
    for (int rank = 0; rank < state.mpiSize; ++rank) {
        size_t count = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
        size_t begin = base * static_cast<size_t>(rank) +
                       std::min(static_cast<size_t>(rank), extra);
        state.rowCounts[rank] = static_cast<int>(count);
        state.rowDispls[rank] = static_cast<int>(begin);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
    CUDA_CHECK(cudaHostRegister(state.radB.data(), state.radB.size() * sizeof(val_t),
                                cudaHostRegisterPortable));

    size_t matrixElements = state.localRows * state.numTriangles;
    CUDA_CHECK(cudaMalloc(&state.dTriangles, state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * state.numTriangles * sizeof(val_t)));
    if (matrixElements != 0) {
        CUDA_CHECK(cudaMalloc(&state.dKij, matrixElements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&state.dTau, matrixElements * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state.dDistances, state.localRows * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMemcpy(state.dTriangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(),
                          state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(),
                          state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * state.numTriangles * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors and delays on GPUs...\n");
    size_t elements = state.localRows * state.numTriangles;
    if (elements == 0) return;
    constexpr unsigned block = 256;
    unsigned grid = static_cast<unsigned>((elements + block - 1) / block);
    precomputeKernel<<<grid, block>>>(state.dTriangles, state.dKij, state.dTau,
                                     state.numTriangles, state.localRowBegin,
                                     state.localRows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    // Fused with computeFormFactors to read triangle geometry only once.
    (void)state;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running distributed GPU wave propagation...\n");

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows != 0) {
            constexpr unsigned block = 256;
            unsigned grid = static_cast<unsigned>((state.localRows + block - 1) / block);
            propagationKernel<<<grid, block>>>(t, state.numTriangles, state.localRowBegin,
                                               state.localRows, state.sourceIndex,
                                               state.numTimesteps / 2, state.dKij,
                                               state.dTau, state.dAreas, state.dRho,
                                               state.dRadB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(state.radB.data() + state.idxTN(t, state.localRowBegin),
                                  state.dRadB + state.idxTN(t, state.localRowBegin),
                                  state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        val_t* slice = state.radB.data() + t * state.numTriangles;
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, slice,
                       state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(state.dRadB + t * state.numTriangles, slice,
                              state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distributed GPU cross-correlations...\n");
    if (state.localRows != 0) {
        constexpr unsigned block = 256;
        unsigned grid = static_cast<unsigned>((state.localRows + block - 1) / block);
        distanceKernel<<<grid, block>>>(state.numTriangles, state.numTimesteps,
                                       state.localRowBegin, state.localRows,
                                       state.sourceIndex, state.dRadB, state.dDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.localRowBegin,
                              state.dDistances, state.localRows * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    bool root = state.mpiRank == 0;
    if (root) printf("\nValidation:\n");

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
            if (root) printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            if (root) printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    if (root) {
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    }

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        if (root) printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    if (root) printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        if (root) printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long localNonZeroKij = 0;
    if (state.localRows != 0) {
        unsigned long long* deviceCount = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceCount, sizeof(*deviceCount)));
        CUDA_CHECK(cudaMemset(deviceCount, 0, sizeof(*deviceCount)));
        size_t elements = state.localRows * state.numTriangles;
        constexpr unsigned block = 256;
        unsigned grid = static_cast<unsigned>((elements + block - 1) / block);
        countPositiveKernel<<<grid, block>>>(state.dKij, elements, deviceCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&localNonZeroKij, deviceCount, sizeof(localNonZeroKij),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceCount));
    }
    unsigned long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    if (root) printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
                     nonZeroKij, state.numTriangles * state.numTriangles,
                     100.0f * static_cast<val_t>(nonZeroKij) /
                         static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        if (root) printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    if (root) printf("  Validation: PASSED\n");
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "roomsim requires at least one CUDA device per node\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

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

    if (targetTriangles <= 0 || timesteps <= 0 || sourceIdx < 0 ||
        reflectivity < 0.0f || reflectivity > 1.0f) {
        if (rank == 0) std::fprintf(stderr, "Invalid problem size, source, or reflectivity\n");
        MPI_Finalize();
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Parallel configuration: %d MPI ranks, %d OpenMP threads/rank, CUDA GPUs\n",
               ranks, omp_get_max_threads());
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = rank;
    state.mpiSize = ranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    long preMax = 0;
    MPI_Reduce(&preDuration, &preMax, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    preDuration = preMax;
    if (rank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    long simMax = 0;
    MPI_Reduce(&simDuration, &simMax, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    simDuration = simMax;
    if (rank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    long distMax = 0;
    MPI_Reduce(&distDuration, &distMax, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    distDuration = distMax;
    if (rank == 0) printf("Distance computation time: %ld ms\n\n", distDuration);

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    if (rank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (rank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    }

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    if (rank == 0) printf("  Per-rank memory usage (rank 0): %.2f MB\n",
                          totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    if (rank == 0) printf("  Result hash: %016lX\n\n", hash);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    bool valid = !validate || validateResults(state);

    CUDA_CHECK(cudaHostUnregister(state.radB.data()));
    CUDA_CHECK(cudaFree(state.dDistances));
    CUDA_CHECK(cudaFree(state.dTau));
    CUDA_CHECK(cudaFree(state.dKij));
    CUDA_CHECK(cudaFree(state.dRadB));
    CUDA_CHECK(cudaFree(state.dRho));
    CUDA_CHECK(cudaFree(state.dAreas));
    CUDA_CHECK(cudaFree(state.dTriangles));
    MPI_Finalize();
    return valid ? 0 : 1;
}
