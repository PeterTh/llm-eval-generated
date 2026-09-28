/**
 * Room Response Simulation Benchmark
 *
 * Hybrid-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy:
 *  - MPI:    row blocks of the Kij matrix and of the distance computation are
 *            distributed across ranks (one GPU per rank), assembled with
 *            MPI_Allgatherv.
 *  - CUDA:   form factors (one thread per triangle pair, iterative octree
 *            traversal), wave propagation (one block per receiver triangle per
 *            timestep), and cross-correlation distances run on the GPU.
 *  - OpenMP: host-side O(N^2) time-delay (Tau) matrix and other host loops.
 *
 * The Monte-Carlo visibility sampling uses a counter-based RNG seeded per
 * triangle pair, so results are deterministic and independent of the number of
 * ranks / GPUs used.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

static int g_mpiRank = 0;
static int g_mpiSize = 1;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
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
};

// ============================================================================
// Flattened Octree (GPU representation)
// ============================================================================

struct GpuOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];   // Node indices, -1 if absent
    int triStart;      // Offset into triangle index array
    int triCount;      // Number of triangle indices (leaf iff > 0)
};

struct FlatOctree {
    std::vector<GpuOctreeNode> nodes;
    std::vector<int> triIndices;
};

static int flattenOctreeNode(const Octree& node, FlatOctree& flat) {
    int id = static_cast<int>(flat.nodes.size());
    flat.nodes.emplace_back();

    GpuOctreeNode gn;
    gn.center = node.center;
    gn.halfExtent = node.halfExtent;
    gn.triStart = static_cast<int>(flat.triIndices.size());
    gn.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) {
        flat.triIndices.push_back(static_cast<int>(idx));
    }
    for (int c = 0; c < 8; ++c) {
        gn.children[c] = node.children[c] ? flattenOctreeNode(*node.children[c], flat) : -1;
    }

    flat.nodes[id] = gn;
    return id;
}

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
// Random Number Generation (counter-based, deterministic per triangle pair)
// ============================================================================

// A stateless splitmix64-based generator: seeded per (i, j) pair so that form
// factor sampling is fully deterministic regardless of the parallel
// decomposition (rank count, block size, thread order).
struct GpuRng {
    uint64_t state;

    __host__ __device__ explicit GpuRng(uint64_t seed) : state(seed * 0x9e3779b97f4a7c15ULL + 0x853c49e6748fea9bULL) {}

    __host__ __device__ val_t rand() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= (z >> 31);
        // 24 high-quality bits -> float in [0, 1)
        return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);
    }
};

// Generate a random point inside a triangle using barycentric coordinates
__host__ __device__ Vec3 randomPointInTriangle(const Triangle& t, GpuRng& rng) {
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

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                               const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, GPU)
// ============================================================================

// Ray vs axis-aligned box test (same slab/cross-term test as the CPU octree)
__device__ bool rayIntersectsBoxDev(const Vec3& p1, const Vec3& p2, const GpuOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal with an explicit stack (depth is bounded by the
// MAX_OCTREE_LEAF_SIZE stopping criterion).
__device__ bool isRayBlockedDev(const Vec3& from, const Vec3& to,
                                const GpuOctreeNode* nodes, const int* triIndices,
                                const Triangle* triangles,
                                int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;  // Root is processed unconditionally, as on the CPU

    while (sp > 0) {
        const GpuOctreeNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {  // Leaf: test triangles directly
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {  // Internal: descend into children whose boxes the ray hits
            for (int c = 0; c < 8; ++c) {
                int child = node.children[c];
                if (child >= 0 && rayIntersectsBoxDev(from, to, nodes[child])) {
                    stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - GPU kernel
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// One thread per (i, j) triangle pair; each thread shoots NUM_RAYS visibility
// rays through the octree and accumulates the Monte-Carlo form factor.
__global__ void kijKernel(const Triangle* triangles, int n,
                          const GpuOctreeNode* nodes, const int* triIndices,
                          long long pairStart, long long pairCount,
                          val_t* kijOut) {
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < pairCount; p += stride) {
        long long pair = pairStart + p;
        int i = static_cast<int>(pair / n);
        int j = static_cast<int>(pair % n);

        val_t kij = ZERO;

        if (i != j) {
            const Triangle triI = triangles[i];
            const Triangle triJ = triangles[j];

            // Cull triangles facing the same direction
            if (triI.normal().dot(triJ.normal()) <= 0.99f) {
                // Seed depends only on (i, j): decomposition-independent
                GpuRng rng(static_cast<uint64_t>(i) * static_cast<uint64_t>(n) +
                           static_cast<uint64_t>(j));

                for (int r = 0; r < NUM_RAYS; ++r) {
                    Vec3 pI = randomPointInTriangle(triI, rng);
                    Vec3 pJ = randomPointInTriangle(triJ, rng);

                    if (isRayBlockedDev(pI, pJ, nodes, triIndices, triangles, i, j)) continue;

                    Vec3 v = pJ - pI;
                    val_t distSqr = v.squaredNorm();
                    if (distSqr < EPSILON) continue;

                    val_t cosPhiI = cosPhiDev(v, triI.normal());
                    val_t cosPhiJ = cosPhiDev(-v, triJ.normal());

                    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                    kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
                }

                kij *= INV_NUM_RAYS;
            }
        }

        kijOut[pair] = kij;
    }
}

// ============================================================================
// Wave Propagation - GPU kernel (one block per receiver triangle, one launch
// per timestep; the timestep loop is inherently sequential because radB at
// time t depends on radB at earlier times)
// ============================================================================

constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(int t, int n,
                              const val_t* kij, const int* tau,
                              const val_t* areas, const val_t* rho,
                              const val_t* radE, val_t* radB) {
    int i = blockIdx.x;
    __shared__ val_t partial[SIM_BLOCK];

    const val_t* kijRow = kij + static_cast<size_t>(i) * n;
    const int* tauRow = tau + static_cast<size_t>(i) * n;

    val_t sumB = ZERO;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tauRow[j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        val_t k = kijRow[j];
        if (k <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        val_t radJ = radB[static_cast<size_t>(t - tauij) * n + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    partial[threadIdx.x] = sumB;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(s)) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Update radiosity: reflection + emission
        size_t ti = static_cast<size_t>(t) * n + i;
        radB[ti] = rho[i] * partial[0] + radE[ti];
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation) - GPU kernel, one thread per
// triangle
// ============================================================================

__global__ void distanceKernel(int n, int numTimesteps, const val_t* radB,
                               int sourceIndex, int iStart, int iCount,
                               val_t* distances) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= iCount) return;
    int i = iStart + p;

    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;

        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    // Device-side buffers (per-rank GPU)
    Triangle* dTriangles = nullptr;
    GpuOctreeNode* dNodes = nullptr;
    int* dTriIndices = nullptr;
    val_t* dKij = nullptr;
    int* dTau = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadE = nullptr;
    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;
};

// Block row partition of [0, n) across MPI ranks
static void rowRange(size_t n, int rank, int size, size_t& start, size_t& count) {
    start = n * static_cast<size_t>(rank) / size;
    size_t end = n * (static_cast<size_t>(rank) + 1) / size;
    count = end - start;
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

    if (g_mpiRank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (g_mpiRank == 0) printf("Building octree...\n");
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
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Flatten octree and upload static data to this rank's GPU
    FlatOctree flat;
    flattenOctreeNode(state.octree, flat);

    size_t n = state.numTriangles;
    CUDA_CHECK(cudaMalloc(&state.dTriangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, flat.nodes.size() * sizeof(GpuOctreeNode)));
    CUDA_CHECK(cudaMalloc(&state.dTriIndices, std::max<size_t>(1, flat.triIndices.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dKij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTriangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, flat.nodes.data(),
                          flat.nodes.size() * sizeof(GpuOctreeNode), cudaMemcpyHostToDevice));
    if (!flat.triIndices.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dTriIndices, flat.triIndices.data(),
                              flat.triIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(),
                          timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * n * sizeof(val_t)));
}

void releaseGpuBuffers(SimulationState& state) {
    cudaFree(state.dTriangles);
    cudaFree(state.dNodes);
    cudaFree(state.dTriIndices);
    cudaFree(state.dKij);
    cudaFree(state.dTau);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dDistances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Each rank computes a block of Kij rows on its GPU; the full matrix is then
// assembled on every rank with MPI_Allgatherv and re-uploaded for the
// (replicated) simulation phase.
void computeFormFactors(SimulationState& state) {
    if (g_mpiRank == 0) printf("Computing form factors (Kij) on %d GPU rank(s)...\n", g_mpiSize);

    size_t n = state.numTriangles;
    size_t rowStart, rowCount;
    rowRange(n, g_mpiRank, g_mpiSize, rowStart, rowCount);

    long long pairStart = static_cast<long long>(rowStart) * n;
    long long pairCount = static_cast<long long>(rowCount) * n;

    if (pairCount > 0) {
        int block = 128;
        long long grid = (pairCount + block - 1) / block;
        if (grid > 1048576) grid = 1048576;
        kijKernel<<<static_cast<unsigned>(grid), block>>>(
            state.dTriangles, static_cast<int>(n), state.dNodes, state.dTriIndices,
            pairStart, pairCount, state.dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.kij.data() + pairStart, state.dKij + pairStart,
                              pairCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    // Assemble the full Kij matrix on all ranks
    std::vector<int> counts(g_mpiSize), displs(g_mpiSize);
    for (int r = 0; r < g_mpiSize; ++r) {
        size_t rs, rc;
        rowRange(n, r, g_mpiSize, rs, rc);
        counts[r] = static_cast<int>(rc * n);
        displs[r] = static_cast<int>(rs * n);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.kij.data(), counts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);

    // Every rank needs the full matrix on its GPU for the simulation phase
    CUDA_CHECK(cudaMemcpy(state.dKij, state.kij.data(), n * n * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    if (g_mpiRank == 0) printf("  Progress: %zu/%zu triangles\n", n, n);
}

void computeTimeDelays(SimulationState& state) {
    if (g_mpiRank == 0) printf("Computing time delays (Tau) with %d OpenMP threads...\n",
                               omp_get_max_threads());

    size_t n = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }

    CUDA_CHECK(cudaMemcpy(state.dTau, state.tau.data(), n * n * sizeof(int),
                          cudaMemcpyHostToDevice));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// The timestep recurrence is sequential, so each rank runs the full (cheap,
// O(N^2) per step) propagation on its own GPU; the result is bitwise identical
// on every rank since all inputs (Kij, Tau) are identical.
void runSimulation(SimulationState& state) {
    if (g_mpiRank == 0) printf("Running wave propagation simulation on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<n, SIM_BLOCK>>>(static_cast<int>(t), n,
                                        state.dKij, state.dTau,
                                        state.dAreas, state.dRho,
                                        state.dRadE, state.dRadB);
        CUDA_CHECK(cudaGetLastError());

        if (g_mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    // Bring radiosity back to the host for validation / reporting
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB,
                          state.numTimesteps * state.numTriangles * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// Rows (triangles) are partitioned across ranks; each rank correlates its
// block on the GPU and results are assembled with MPI_Allgatherv.
void computeDistances(SimulationState& state) {
    if (g_mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    size_t n = state.numTriangles;
    size_t iStart, iCount;
    rowRange(n, g_mpiRank, g_mpiSize, iStart, iCount);

    if (iCount > 0) {
        int block = 128;
        int grid = static_cast<int>((iCount + block - 1) / block);
        distanceKernel<<<grid, block>>>(static_cast<int>(n),
                                        static_cast<int>(state.numTimesteps),
                                        state.dRadB, static_cast<int>(state.sourceIndex),
                                        static_cast<int>(iStart), static_cast<int>(iCount),
                                        state.dDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + iStart, state.dDistances + iStart,
                              iCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(g_mpiSize), displs(g_mpiSize);
    for (int r = 0; r < g_mpiSize; ++r) {
        size_t rs, rc;
        rowRange(n, r, g_mpiSize, rs, rc);
        counts[r] = static_cast<int>(rc);
        displs[r] = static_cast<int>(rs);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), counts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
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
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
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
    long long nonZeroKij = 0;
    #pragma omp parallel for schedule(static) reduction(+:nonZeroKij)
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpiSize);

    // Bind each rank to a GPU (round-robin over the visible devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_mpiRank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(g_mpiRank % deviceCount));

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
            if (g_mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs: %d, OpenMP threads: %d\n",
               g_mpiSize, deviceCount, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (g_mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (g_mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (g_mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    int exitCode = 0;

    if (g_mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
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
                exitCode = 1;
            }
        }
    }

    releaseGpuBuffers(state);

    // All ranks return the same exit code
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
