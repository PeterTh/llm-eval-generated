/**
 * Room Response Simulation Benchmark (hybrid MPI + OpenMP + CUDA)
 *
 * Radiosity-based room impulse response simulation:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * Parallelization:
 *  - MPI: the triangle index space is block-partitioned across ranks; each rank
 *    owns a contiguous range of rows of the Kij/Tau matrices and of the
 *    radiosity update. Per-timestep radiosity rows and the final distances are
 *    exchanged with MPI_Allgatherv. One GPU is assigned per rank.
 *  - CUDA: form-factor Monte Carlo ray tracing (octree flattened into arrays
 *    and traversed with an explicit stack), the per-timestep radiosity update,
 *    and the cross-correlation distance search all run as GPU kernels.
 *  - OpenMP: host-side O(N^2) work (time-delay matrix, validation scans).
 *
 * Monte Carlo sampling uses a deterministic counter-based RNG keyed on the
 * (i, j) triangle pair, so results are identical regardless of the number of
 * ranks or threads.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
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

static int g_rank = 0;
static int g_nprocs = 1;

#define ROOT_PRINTF(...) do { if (g_rank == 0) printf(__VA_ARGS__); } while (0)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
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
// Triangle-Box Overlap Test (for octree construction, host only)
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
// Octree for Spatial Acceleration (built on host, flattened for the GPU)
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

// ----------------------------------------------------------------------------
// Flattened (GPU-friendly) octree representation
// ----------------------------------------------------------------------------

struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];    // node indices, -1 if absent
    int triStart;    // offset into the leaf triangle-index array
    int triCount;    // 0 for internal nodes
};

struct DevTri {
    Vec3 a, b, c;
    Vec3 n;
};

static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<int>& leafTris) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    DevNode dn;
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    if (!node.triangleIndices.empty()) {
        dn.triStart = static_cast<int>(leafTris.size());
        dn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t t : node.triangleIndices) leafTris.push_back(static_cast<int>(t));
        for (int i = 0; i < 8; ++i) dn.child[i] = -1;
    } else {
        dn.triStart = 0;
        dn.triCount = 0;
        for (int i = 0; i < 8; ++i) {
            dn.child[i] = node.children[i] ? flattenOctree(*node.children[i], nodes, leafTris) : -1;
        }
    }

    nodes[myIdx] = dn;
    return myIdx;
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
// Deterministic counter-based RNG (rank/thread-count independent)
// ============================================================================

__host__ __device__ inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__host__ __device__ inline val_t rand01(uint64_t& s) {
    // 24 high-quality bits -> uniform float in [0, 1)
    return static_cast<val_t>(splitmix64(s) >> 40) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ inline Vec3 randomPointInTriangle(const DevTri& t, uint64_t& s) {
    val_t u = rand01(s);
    val_t v = rand01(s);
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

__device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (flattened-octree traversal on the GPU)
// ============================================================================

// Check if a ray intersects a node's bounding box
__device__ inline bool rayIntersectsBox(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > n.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > n.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > n.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.halfExtent.y * ad.z + n.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.halfExtent.z * ad.x + n.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.halfExtent.x * ad.y + n.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevNode* nodes, const int* leafTris, const DevTri* tris,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;  // root (checked unconditionally, like the recursive version)

    while (sp > 0) {
        const DevNode& n = nodes[stack[--sp]];

        if (n.triCount > 0) {
            // Leaf: test triangles directly
            for (int k = 0; k < n.triCount; ++k) {
                int idx = leafTris[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DevTri& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;  // Ray is blocked
            }
        } else {
            // Internal: descend into children whose box the ray touches
            for (int i = 0; i < 8; ++i) {
                int ci = n.child[i];
                if (ci >= 0 && rayIntersectsBox(nodes[ci], from, to)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) — GPU kernel
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// One block per local row i; threads stride over columns j.
__global__ void kijKernel(int rowStart, int N,
                          const DevTri* __restrict__ tris,
                          const DevNode* __restrict__ nodes,
                          const int* __restrict__ leafTris,
                          val_t* __restrict__ kijLocal) {
    const int li = blockIdx.x;
    const int i = rowStart + li;
    const DevTri triI = tris[i];

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        val_t kij = ZERO;

        // Cull self-pairs and triangles facing the same direction
        if (j != i && triI.n.dot(tris[j].n) <= 0.99f) {
            const DevTri triJ = tris[j];

            // Deterministic per-pair RNG stream
            uint64_t rngState = (static_cast<uint64_t>(i) * static_cast<uint64_t>(N) +
                                 static_cast<uint64_t>(j)) ^ 0x5DEECE66D2026900ULL;

            for (int r = 0; r < NUM_RAYS; ++r) {
                Vec3 pI = randomPointInTriangle(triI, rngState);
                Vec3 pJ = randomPointInTriangle(triJ, rngState);

                if (isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) continue;

                Vec3 v = pJ - pI;
                val_t distSqr = v.squaredNorm();
                if (distSqr < EPSILON) continue;

                val_t cosPhiI = cosPhi(v, triI.n);
                val_t cosPhiJ = cosPhi(-v, triJ.n);

                if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
            kij *= INV_NUM_RAYS;
        }

        kijLocal[static_cast<size_t>(li) * N + j] = kij;
    }
}

// Fold the per-column area clamp into the form factors:
// w[i][j] = min(kij[i][j] * area[j], 1)
__global__ void weightKernel(size_t count, int N,
                             const val_t* __restrict__ areas,
                             val_t* __restrict__ kijLocal) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int j = static_cast<int>(idx % N);
    kijLocal[idx] = fminf(kijLocal[idx] * areas[j], ONE);
}

// ============================================================================
// Simulation Phase (Wave Propagation) — GPU kernel, one timestep per launch
// ============================================================================

constexpr int SIM_BLOCK = 256;

// One block per local row i; the reduction order over j is fixed by SIM_BLOCK,
// so results do not depend on the number of ranks.
__global__ void simStepKernel(int t, int rowStart, int N,
                              const val_t* __restrict__ wLocal,
                              const int* __restrict__ tauLocal,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* __restrict__ radB) {
    const int li = blockIdx.x;
    const int i = rowStart + li;
    const val_t* wRow = wLocal + static_cast<size_t>(li) * N;
    const int* tauRow = tauLocal + static_cast<size_t>(li) * N;

    __shared__ val_t sh[SIM_BLOCK];

    val_t sum = ZERO;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tauRow[j];
        if (t < tauij) continue;  // Wave hasn't yet propagated from j to i

        // Form factor weight * source radiosity at emission time
        sum += wRow[j] * radB[static_cast<size_t>(t - tauij) * N + j];
    }

    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int off = SIM_BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Update radiosity: reflection + emission
        radB[static_cast<size_t>(t) * N + i] = rho[i] * sh[0] + radE[static_cast<size_t>(t) * N + i];
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation) — GPU kernel
// ============================================================================

// One thread per local triangle; loops match the sequential evaluation order.
__global__ void distKernel(int rowStart, int rowsLocal, int N, int T, int srcIdx,
                           const val_t* __restrict__ radB,
                           val_t* __restrict__ distLocal) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= rowsLocal) return;
    int i = rowStart + li;

    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = 0; t < T; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * N + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * N + srcIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distLocal[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Tau (time delay) Computation (host, OpenMP)
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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, host copy)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, host copy)
    std::vector<val_t> distances;   // Computed distances from source (full, gathered)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI row decomposition (contiguous block of triangle rows per rank)
    int rowStart = 0;
    int rowsLocal = 0;
    std::vector<int> rowCounts;     // per-rank row counts
    std::vector<int> rowDispls;     // per-rank row offsets

    // Device buffers (per rank)
    DevTri* dTris = nullptr;
    DevNode* dNodes = nullptr;
    int* dLeafTris = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadE = nullptr;         // full T x N
    val_t* dRadB = nullptr;         // full T x N
    val_t* dKij = nullptr;          // local rows x N (becomes weights after clamp)
    int* dTau = nullptr;            // local rows x N
    val_t* dDist = nullptr;         // local rows

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (deterministic, replicated on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
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
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Row decomposition across MPI ranks
    int N = static_cast<int>(state.numTriangles);
    state.rowCounts.resize(g_nprocs);
    state.rowDispls.resize(g_nprocs);
    int offset = 0;
    for (int r = 0; r < g_nprocs; ++r) {
        state.rowCounts[r] = N / g_nprocs + (r < N % g_nprocs ? 1 : 0);
        state.rowDispls[r] = offset;
        offset += state.rowCounts[r];
    }
    state.rowStart = state.rowDispls[g_rank];
    state.rowsLocal = state.rowCounts[g_rank];

    // Flatten octree and upload static data to this rank's GPU
    std::vector<DevNode> nodes;
    std::vector<int> leafTris;
    flattenOctree(state.octree, nodes, leafTris);

    std::vector<DevTri> tris(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        tris[i] = {state.triangles[i].a, state.triangles[i].b,
                   state.triangles[i].c, state.triangles[i].normal()};
    }

    size_t TN = timesteps * state.numTriangles;
    size_t rowsN = static_cast<size_t>(state.rowsLocal) * state.numTriangles;

    CUDA_CHECK(cudaMalloc(&state.dTris, tris.size() * sizeof(DevTri)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&state.dLeafTris, std::max<size_t>(1, leafTris.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, TN * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, TN * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dKij, std::max<size_t>(1, rowsN) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, std::max<size_t>(1, rowsN) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dDist, std::max<size_t>(1, state.rowsLocal) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTris, tris.data(), tris.size() * sizeof(DevTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, nodes.data(), nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    if (!leafTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dLeafTris, leafTris.data(), leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), TN * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, TN * sizeof(val_t)));
}

void freeDeviceState(SimulationState& state) {
    cudaFree(state.dTris);
    cudaFree(state.dNodes);
    cudaFree(state.dLeafTris);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dKij);
    cudaFree(state.dTau);
    cudaFree(state.dDist);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Returns the number of non-zero form factors across all ranks (for validation)
long long computeFormFactors(SimulationState& state, bool countNonZero) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    int N = static_cast<int>(state.numTriangles);
    if (state.rowsLocal > 0) {
        kijKernel<<<state.rowsLocal, 128>>>(state.rowStart, N,
                                            state.dTris, state.dNodes, state.dLeafTris,
                                            state.dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Count non-zero form factors before clamping into weights (validation only)
    long long nonZeroKij = 0;
    if (countNonZero) {
        size_t rowsN = static_cast<size_t>(state.rowsLocal) * N;
        std::vector<val_t> kijHost(rowsN);
        if (rowsN > 0) {
            CUDA_CHECK(cudaMemcpy(kijHost.data(), state.dKij, rowsN * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        long long localCount = 0;
        #pragma omp parallel for reduction(+:localCount) schedule(static)
        for (size_t k = 0; k < rowsN; ++k) {
            if (kijHost[k] > EPSILON) localCount++;
        }
        MPI_Allreduce(&localCount, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    // Fold the area clamp into the stored factors: w = min(kij * area_j, 1)
    size_t count = static_cast<size_t>(state.rowsLocal) * N;
    if (count > 0) {
        int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        weightKernel<<<blocks, threads>>>(count, N, state.dAreas, state.dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return nonZeroKij;
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");

    int N = static_cast<int>(state.numTriangles);
    std::vector<int> tauLocal(static_cast<size_t>(state.rowsLocal) * N, 0);

    #pragma omp parallel for schedule(static)
    for (int li = 0; li < state.rowsLocal; ++li) {
        size_t i = static_cast<size_t>(state.rowStart + li);
        for (int j = 0; j < N; ++j) {
            if (static_cast<size_t>(j) == i) continue;
            tauLocal[static_cast<size_t>(li) * N + j] =
                computeTau(state.triangles[i], state.triangles[j]);
        }
    }

    if (!tauLocal.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dTau, tauLocal.data(), tauLocal.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    int N = static_cast<int>(state.numTriangles);
    std::vector<val_t> rowFull(N);
    std::vector<val_t> rowLocal(std::max(1, state.rowsLocal));

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Each rank updates radB[t] for its own rows on the GPU. Because every
        // tau_ij >= 1 for i != j, all rows within a timestep are independent.
        if (state.rowsLocal > 0) {
            simStepKernel<<<state.rowsLocal, SIM_BLOCK>>>(static_cast<int>(t), state.rowStart, N,
                                                          state.dKij, state.dTau, state.dRho,
                                                          state.dRadE, state.dRadB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(rowLocal.data(),
                                  state.dRadB + t * state.numTriangles + state.rowStart,
                                  state.rowsLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        // Exchange the freshly computed timestep row with all ranks
        MPI_Allgatherv(rowLocal.data(), state.rowsLocal, MPI_FLOAT,
                       rowFull.data(), state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(state.dRadB + t * state.numTriangles, rowFull.data(),
                              N * sizeof(val_t), cudaMemcpyHostToDevice));
        std::memcpy(&state.radB[state.idxTN(t, 0)], rowFull.data(), N * sizeof(val_t));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);

    std::vector<val_t> distLocal(std::max(1, state.rowsLocal));
    if (state.rowsLocal > 0) {
        int threads = 128;
        int blocks = (state.rowsLocal + threads - 1) / threads;
        distKernel<<<blocks, threads>>>(state.rowStart, state.rowsLocal, N, T,
                                        static_cast<int>(state.sourceIndex),
                                        state.dRadB, state.dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(distLocal.data(), state.dDist,
                              state.rowsLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    MPI_Allgatherv(distLocal.data(), state.rowsLocal, MPI_FLOAT,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (runs on rank 0; nonZeroKij is pre-reduced across ranks)
// ============================================================================

bool validateResults(const SimulationState& state, long long nonZeroKij) {
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
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
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
    ROOT_PRINTF("Usage: %s [options]\n", progName);
    ROOT_PRINTF("Options:\n");
    ROOT_PRINTF("  -n <num>     Target number of triangles (default: 320)\n");
    ROOT_PRINTF("               Actual count will be rounded to nearest icosphere level:\n");
    ROOT_PRINTF("               20, 80, 320, 1280, 5120, 20480\n");
    ROOT_PRINTF("  -t <num>     Number of timesteps (default: 50)\n");
    ROOT_PRINTF("  -s <num>     Source triangle index (default: 0)\n");
    ROOT_PRINTF("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    ROOT_PRINTF("  -v           Enable validation\n");
    ROOT_PRINTF("  -o           Print results for external validation\n");
    ROOT_PRINTF("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    // Assign one GPU per rank (round-robin over the node's devices)
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(g_rank % devCount));

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
            ROOT_PRINTF("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("Parallelization: MPI ranks=%d, OpenMP threads=%d, CUDA devices=%d\n",
                g_nprocs, omp_get_max_threads(), devCount);
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    long long nonZeroKij = computeFormFactors(state, validate);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", t);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");

    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int ok = 1;
    if (validate) {
        if (g_rank == 0) {
            ok = validateResults(state, nonZeroKij) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    freeDeviceState(state);
    MPI_Finalize();
    return ok ? 0 : 1;
}
