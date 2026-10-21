/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy:
 *   - MPI partitions the triangle rows of the Kij/Tau matrices, the radiosity
 *     update and the distance estimation across cluster ranks. Each rank owns a
 *     contiguous block of output triangles, so no large matrix ever has to be
 *     exchanged. Only the (T x N) radiosity rows are all-gathered per timestep.
 *   - OpenMP parallelizes the host-side loops (time-delay matrix, area setup,
 *     validation reductions).
 *   - CUDA executes the compute-heavy kernels (visibility/form factors,
 *     radiosity propagation, cross-correlation) on the local accelerator.
 *
 * The Monte-Carlo samples are reproduced from the original mt19937 stream:
 * each rank jumps to the exact stream position of its first pair (the number
 * of preceding draws is known from the normal-culling test) and streams the
 * samples to the GPU. The form factors therefore match the sequential
 * reference implementation, independent of the parallel schedule. The
 * radiosity and cross-correlation recurrence keeps the original term order;
 * only the warp-level reduction tree differs.
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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
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
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
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
        val_t rr = boxHalfSize.x * std::abs(axis.x) +
                   boxHalfSize.y * std::abs(axis.y) +
                   boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > rr || maxP < -rr);
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
// Flattened GPU Octree Node
// ============================================================================

struct GpuNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int firstTri;
    int triCount;
};

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

    // Flatten the tree into a contiguous node array suitable for the GPU.
    // Leaf triangle indices are packed into a single flat array.
    void flatten(std::vector<GpuNode>& nodes, std::vector<idx_t>& leafTris) const {
        flattenNode(nodes, leafTris);
    }

private:
    int flattenNode(std::vector<GpuNode>& nodes, std::vector<idx_t>& leafTris) const {
        int myIndex = static_cast<int>(nodes.size());
        nodes.push_back(GpuNode{});

        GpuNode node{};
        node.center = center;
        node.halfExtent = halfExtent;
        for (int i = 0; i < 8; ++i) node.child[i] = -1;
        node.firstTri = 0;
        node.triCount = 0;

        if (!triangleIndices.empty()) {
            node.firstTri = static_cast<int>(leafTris.size());
            node.triCount = static_cast<int>(triangleIndices.size());
            for (size_t idx : triangleIndices) leafTris.push_back(static_cast<idx_t>(idx));
        } else {
            for (int i = 0; i < 8; ++i) {
                if (children[i]) {
                    node.child[i] = children[i]->flattenNode(nodes, leafTris);
                }
            }
        }

        nodes[myIndex] = node;
        return myIndex;
    }

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
// Device-side ray tracing helpers
// ============================================================================

// Check if a ray intersects a node's bounding box
__device__ __forceinline__ bool devRayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                                     const GpuNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Möller-Trumbore ray-triangle intersection
__device__ __forceinline__ val_t devRayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                          const Vec3& v0, const Vec3& v1,
                                                          const Vec3& v2) {
    constexpr val_t FLT_MAX_VAL = 3.4028234663852886e38f;

    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX_VAL;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX_VAL;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_VAL;

    return e2.dot(qvec) * invDet;
}

// Check if a ray between two triangles is blocked by any other triangle
__device__ __forceinline__ bool devIsRayBlocked(const Vec3& from, const Vec3& to,
                                                 const Triangle* tris,
                                                 const GpuNode* nodes,
                                                 const idx_t* leafTris,
                                                 int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        const GpuNode& node = nodes[ni];
        if (!devRayIntersectsBox(from, to, node)) continue;

        if (node.triCount > 0) {
            int end = node.firstTri + node.triCount;
            for (int k = node.firstTri; k < end; ++k) {
                int idx = static_cast<int>(leafTris[k]);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = tris[idx];
                val_t dist = devRayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int ci = 0; ci < 8; ++ci) {
                int ch = node.child[ci];
                if (ch >= 0 && sp < 128) stack[sp++] = ch;
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Compute form factor Kij for one (i, j) pair per thread. The random samples
// are generated on the host from the original mt19937 stream and passed in
// through randOffsets/randVals, so the numerical result is identical to the
// sequential reference implementation.
__global__ void kijKernel(const Triangle* __restrict__ tris,
                          const GpuNode* __restrict__ nodes,
                          const idx_t* __restrict__ leafTris,
                          const int* __restrict__ randOffsets,
                          const val_t* __restrict__ randVals,
                          int numTriangles,
                          int rowStart,
                          val_t* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int li = blockIdx.y;
    if (j >= numTriangles) return;

    int i = rowStart + li;
    size_t oi = static_cast<size_t>(li) * numTriangles + j;
    int seedOffset = randOffsets[static_cast<size_t>(li) * numTriangles + j];

    if (seedOffset < 0) {
        out[oi] = ZERO;
        return;
    }

    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];

    Vec3 abI = triI.b - triI.a;
    Vec3 acI = triI.c - triI.a;
    Vec3 abJ = triJ.b - triJ.a;
    Vec3 acJ = triJ.c - triJ.a;

    const val_t* rv = randVals + seedOffset;
    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t u = rv[4 * r + 0];
        val_t v = rv[4 * r + 1];
        if (u + v > ONE) {
            u = ONE - u;
            v = ONE - v;
        }
        Vec3 pI = triI.a + abI * u + acI * v;

        u = rv[4 * r + 2];
        v = rv[4 * r + 3];
        if (u + v > ONE) {
            u = ONE - u;
            v = ONE - v;
        }
        Vec3 pJ = triJ.a + abJ * u + acJ * v;

        if (devIsRayBlocked(pI, pJ, tris, nodes, leafTris, i, j)) continue;

        Vec3 vv = pJ - pI;
        val_t distSqr = vv.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t vNorm = sqrtf(distSqr);
        if (vNorm <= EPSILON) continue;

        val_t cosPhiI = vv.dot(triI._normal) / vNorm;
        if (cosPhiI < ZERO) cosPhiI = ZERO;
        val_t cosPhiJ = (-vv).dot(triJ._normal) / vNorm;
        if (cosPhiJ < ZERO) cosPhiJ = ZERO;

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    out[oi] = kij * INV_NUM_RAYS;
}

// One warp per output triangle. Dependencies only reach previous timesteps
// (tau >= 1 for distinct triangles), so each timestep is embarrassingly
// parallel. Lanes cover consecutive j values, which makes the row accesses
// coalesced; a warp reduction combines the per-lane partial sums.
__global__ void radiosityKernel(const int* __restrict__ tau,
                                const val_t* __restrict__ kij,
                                const val_t* __restrict__ areas,
                                const val_t* __restrict__ rho,
                                const val_t* __restrict__ radE,
                                const val_t* __restrict__ radB,
                                val_t* __restrict__ out,
                                int numTriangles,
                                int rowStart,
                                int localRows,
                                int t) {
    int li = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (li >= localRows) return;

    int i = rowStart + li;
    const int* tauRow = tau + static_cast<size_t>(li) * numTriangles;
    const val_t* kijRow = kij + static_cast<size_t>(li) * numTriangles;

    val_t sumB = ZERO;
    for (int j = lane; j < numTriangles; j += 32) {
        if (i == j) continue;

        int tauij = tauRow[j];
        if (t < tauij) continue;

        val_t kijVal = kijRow[j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t - tauij);
        val_t radJ = radB[srcTime * static_cast<size_t>(numTriangles) + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijVal * areas[j], ONE) * radJ;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumB += __shfl_down_sync(0xffffffffu, sumB, offset);
    }

    if (lane == 0) {
        out[li] = rho[i] * sumB + radE[static_cast<size_t>(t) * numTriangles + i];
    }
}

// One warp per output triangle: discrete cross-correlation. Each lane owns a
// disjoint set of lags and keeps the exact per-lag summation order; the warp
// reduction selects the largest correlation and, on ties, the smallest lag,
// which matches the sequential strict-greater-than update.
__global__ void distanceKernel(const val_t* __restrict__ radB,
                               val_t* __restrict__ out,
                               int numTriangles,
                               int numTimesteps,
                               int rowStart,
                               int localRows,
                               int sourceIndex) {
    int li = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (li >= localRows) return;

    int i = rowStart + li;
    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = lane; t < numTimesteps; t += 32) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_t otherCorr = __shfl_down_sync(0xffffffffu, maxCorr, offset);
        int otherT = __shfl_down_sync(0xffffffffu, bestT, offset);
        if (otherCorr > maxCorr || (otherCorr == maxCorr && otherT < bestT)) {
            maxCorr = otherCorr;
            bestT = otherT;
        }
    }

    if (lane == 0) {
        out[li] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
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
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Random Number Generation (identical to the sequential reference)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed) {}
    explicit RandomGenerator(const std::mt19937& engine) : rng(engine) {}
    // Exactly matches std::uniform_real_distribution<float>(0, 1) applied to
    // std::mt19937 (one raw 32-bit draw per sample), but is cheaper.
    val_t rand() { return static_cast<val_t>(rng()) * (1.0f / 4294967296.0f); }
    void discard(unsigned long long n) { rng.discard(n); }
    std::mt19937 engine() const { return rng; }
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Local row block of form factors (localRows x N)
    std::vector<int> tau;           // Local row block of time delays (localRows x N)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI row partition
    int rank = 0;
    int nranks = 1;
    size_t localStart = 0;
    size_t localRows = 0;
    std::vector<int> counts;
    std::vector<int> displs;

    long long nonZeroKijGlobal = 0;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Device Buffers
// ============================================================================

struct DeviceBuffers {
    Triangle* tris = nullptr;
    GpuNode* nodes = nullptr;
    idx_t* leafTris = nullptr;
    val_t* area = nullptr;
    val_t* rho = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    int* tau = nullptr;
    val_t* kij = nullptr;
    val_t* simOut = nullptr;
    val_t* distOut = nullptr;
    int* randOffsets = nullptr;
    val_t* randVals = nullptr;
};

constexpr size_t KIJ_CHUNK_ROWS = 128;

static int g_rank = 0;

void setupDevice(const SimulationState& state, DeviceBuffers& dev) {
    size_t N = state.numTriangles;
    size_t TN = state.numTimesteps * N;
    size_t localElems = state.localRows * N;

    CUDA_CHECK(cudaMalloc(&dev.tris, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(dev.tris, state.triangles.data(), N * sizeof(Triangle),
                          cudaMemcpyHostToDevice));

    std::vector<GpuNode> hostNodes;
    std::vector<idx_t> hostLeafTris;
    state.octree.flatten(hostNodes, hostLeafTris);

    if (!hostNodes.empty()) {
        CUDA_CHECK(cudaMalloc(&dev.nodes, hostNodes.size() * sizeof(GpuNode)));
        CUDA_CHECK(cudaMemcpy(dev.nodes, hostNodes.data(), hostNodes.size() * sizeof(GpuNode),
                              cudaMemcpyHostToDevice));
    }
    if (!hostLeafTris.empty()) {
        CUDA_CHECK(cudaMalloc(&dev.leafTris, hostLeafTris.size() * sizeof(idx_t)));
        CUDA_CHECK(cudaMemcpy(dev.leafTris, hostLeafTris.data(), hostLeafTris.size() * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&dev.area, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(dev.area, state.areas.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev.rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(dev.rho, state.rho.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    if (TN > 0) {
        CUDA_CHECK(cudaMalloc(&dev.radE, TN * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(dev.radE, state.radE.data(), TN * sizeof(val_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&dev.radB, TN * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.radB, 0, TN * sizeof(val_t)));
    }

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&dev.tau, localElems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dev.kij, localElems * sizeof(val_t)));
    }
    if (state.localRows > 0) {
        CUDA_CHECK(cudaMalloc(&dev.simOut, state.localRows * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dev.distOut, state.localRows * sizeof(val_t)));
    }

    size_t chunkRows = std::min(KIJ_CHUNK_ROWS, state.localRows);
    if (chunkRows > 0 && N > 0) {
        size_t chunkPairs = chunkRows * N;
        CUDA_CHECK(cudaMalloc(&dev.randOffsets, chunkPairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dev.randVals, chunkPairs * 4 * NUM_RAYS * sizeof(val_t)));
    }
}

void freeDevice(DeviceBuffers& dev) {
    cudaFree(dev.tris);
    cudaFree(dev.nodes);
    cudaFree(dev.leafTris);
    cudaFree(dev.area);
    cudaFree(dev.rho);
    cudaFree(dev.radE);
    cudaFree(dev.radB);
    cudaFree(dev.tau);
    cudaFree(dev.kij);
    cudaFree(dev.simOut);
    cudaFree(dev.distOut);
    cudaFree(dev.randOffsets);
    cudaFree(dev.randVals);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int nranks) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.rank = rank;
    state.nranks = nranks;

    size_t N = state.numTriangles;
    state.localStart = (N * static_cast<size_t>(rank)) / static_cast<size_t>(nranks);
    size_t localEnd = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(nranks);
    state.localRows = localEnd - state.localStart;

    state.counts.resize(nranks);
    state.displs.resize(nranks);
    for (int r = 0; r < nranks; ++r) {
        size_t lo = (N * static_cast<size_t>(r)) / static_cast<size_t>(nranks);
        size_t hi = (N * static_cast<size_t>(r + 1)) / static_cast<size_t>(nranks);
        state.counts[r] = static_cast<int>(hi - lo);
        state.displs[r] = static_cast<int>(lo);
    }

    if (g_rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", N);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas with OpenMP
    state.areas.resize(N);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Initialize matrices (only the local row block of Kij/Tau is stored)
    state.kij.assign(state.localRows * N, ZERO);
    state.tau.assign(state.localRows * N, 0);
    state.radE.assign(timesteps * N, ZERO);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOff = timesteps / 2;
    #pragma omp parallel for schedule(static)
    for (long long t = 0; t < static_cast<long long>(timeOff); ++t) {
        state.radE[state.idxTN(static_cast<size_t>(t), state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, DeviceBuffers& dev) {
    if (g_rank == 0) printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;
    size_t start = state.localStart;
    std::vector<Triangle>& triangles = state.triangles;
    std::vector<int>& tau = state.tau;

    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(state.localRows); ++li) {
        size_t i = start + static_cast<size_t>(li);
        Vec3 ci = triangles[i].center();
        int* row = tau.data() + static_cast<size_t>(li) * N;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                row[j] = 0;
            } else {
                Vec3 cj = triangles[j].center();
                val_t dist = (ci - cj).norm();
                row[j] = static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
            }
        }
    }

    if (!state.tau.empty()) {
        CUDA_CHECK(cudaMemcpy(dev.tau, state.tau.data(), state.tau.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
}

void computeFormFactors(SimulationState& state, DeviceBuffers& dev) {
    if (g_rank == 0) printf("Computing form factors (Kij)...\n");

    size_t N = state.numTriangles;
    int nInt = static_cast<int>(N);

    // Number of Monte-Carlo pairs (and therefore mt19937 draws) that the
    // sequential reference consumes before this rank's first row. Only pairs
    // whose normals are not almost parallel consume 4 * NUM_RAYS draws.
    long long prefixPairs = 0;
    #pragma omp parallel for reduction(+:prefixPairs) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.localStart); ++i) {
        const Vec3& ni = state.triangles[static_cast<size_t>(i)]._normal;
        long long count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (static_cast<size_t>(i) == j) continue;
            if (ni.dot(state.triangles[j]._normal) <= 0.99f) count++;
        }
        prefixPairs += count;
    }

    RandomGenerator baseRng(42);
    baseRng.discard(static_cast<unsigned long long>(prefixPairs) * 4ULL * NUM_RAYS);

    size_t chunkRows = std::min(KIJ_CHUNK_ROWS, state.localRows);
    std::vector<int> hOffsets;
    std::vector<val_t> hVals;
    std::vector<size_t> rowDraws;
    std::vector<size_t> rowPref;
    if (chunkRows > 0 && N > 0) {
        hOffsets.resize(chunkRows * N);
        hVals.resize(chunkRows * N * 4 * NUM_RAYS);
        rowDraws.resize(chunkRows);
        rowPref.resize(chunkRows + 1);
    }

    const size_t drawsPerPair = 4 * NUM_RAYS;
    for (size_t r0 = 0; r0 < state.localRows; r0 += KIJ_CHUNK_ROWS) {
        size_t rows = std::min(KIJ_CHUNK_ROWS, state.localRows - r0);
        size_t pairCount = rows * N;
        size_t valCount = 0;

        if (rows > 0 && N > 0) {
            // Number of draw-consuming pairs per row (used to split the
            // sequential RNG stream across OpenMP threads).
            #pragma omp parallel for schedule(static)
            for (long long lr = 0; lr < static_cast<long long>(rows); ++lr) {
                size_t i = state.localStart + r0 + static_cast<size_t>(lr);
                const Vec3& ni = state.triangles[i]._normal;
                size_t count = 0;
                for (size_t j = 0; j < N; ++j) {
                    if (i == j) continue;
                    if (ni.dot(state.triangles[j]._normal) <= 0.99f) count++;
                }
                rowDraws[static_cast<size_t>(lr)] = count;
            }

            rowPref[0] = 0;
            for (size_t lr = 0; lr < rows; ++lr) {
                rowPref[lr + 1] = rowPref[lr] + rowDraws[lr];
            }
            valCount = rowPref[rows] * drawsPerPair;

            // Each thread jumps to the exact position of its row range in the
            // global mt19937 stream and fills its disjoint output slice.
            // Threads beyond the chunk size would only add jump overhead.
            int genThreads = std::max(1, std::min(omp_get_max_threads(), 32));
            std::mt19937 endEngine;
            #pragma omp parallel num_threads(genThreads)
            {
                int tid = omp_get_thread_num();
                int nt = omp_get_num_threads();
                size_t lrBegin = (rows * static_cast<size_t>(tid)) / static_cast<size_t>(nt);
                size_t lrEnd = (rows * static_cast<size_t>(tid + 1)) / static_cast<size_t>(nt);
                bool isLast = (tid == nt - 1);

                // Threads beyond the row range have no work, except the last
                // thread, which advances the shared stream to the chunk end.
                if (lrBegin < lrEnd || isLast) {
                    RandomGenerator local(baseRng);
                    size_t discardRows = (lrBegin < lrEnd) ? lrBegin : rows;
                    local.discard(static_cast<unsigned long long>(rowPref[discardRows]) * drawsPerPair);

                    for (size_t lr = lrBegin; lr < lrEnd; ++lr) {
                        size_t i = state.localStart + r0 + lr;
                        const Vec3& ni = state.triangles[i]._normal;
                        size_t valIdx = rowPref[lr] * drawsPerPair;
                        for (size_t j = 0; j < N; ++j) {
                            size_t p = lr * N + j;
                            if (i == j || ni.dot(state.triangles[j]._normal) > 0.99f) {
                                hOffsets[p] = -1;
                                continue;
                            }
                            hOffsets[p] = static_cast<int>(valIdx);
                            for (size_t k = 0; k < drawsPerPair; ++k) {
                                hVals[valIdx++] = local.rand();
                            }
                        }
                    }

                    if (isLast) endEngine = local.engine();
                }
            }
            baseRng = RandomGenerator(endEngine);
        }

        CUDA_CHECK(cudaMemcpy(dev.randOffsets, hOffsets.data(),
                              pairCount * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.randVals, hVals.data(),
                              valCount * sizeof(val_t), cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((nInt + 255) / 256, static_cast<unsigned>(rows));
        kijKernel<<<grid, block>>>(dev.tris, dev.nodes, dev.leafTris,
                                   dev.randOffsets, dev.randVals, nInt,
                                   static_cast<int>(state.localStart + r0),
                                   dev.kij + r0 * N);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (!state.kij.empty()) {
        CUDA_CHECK(cudaMemcpy(state.kij.data(), dev.kij, state.kij.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    long long localNonZero = 0;
    #pragma omp parallel for reduction(+:localNonZero) schedule(static)
    for (long long k = 0; k < static_cast<long long>(state.kij.size()); ++k) {
        if (state.kij[k] > EPSILON) localNonZero++;
    }
    MPI_Allreduce(&localNonZero, &state.nonZeroKijGlobal, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, DeviceBuffers& dev) {
    if (g_rank == 0) printf("Running wave propagation simulation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int localRows = static_cast<int>(state.localRows);
    int blocks = (localRows + 7) / 8;  // 8 warps (rows) per 256-thread block
    std::vector<val_t> localOut(localRows > 0 ? state.localRows : 1, ZERO);

    for (int t = 0; t < T; ++t) {
        if (localRows > 0) {
            radiosityKernel<<<blocks, 256>>>(dev.tau, dev.kij, dev.area, dev.rho,
                                             dev.radE, dev.radB, dev.simOut,
                                             N, static_cast<int>(state.localStart),
                                             localRows, t);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localOut.data(), dev.simOut,
                                  state.localRows * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(localOut.data(), localRows, MPI_FLOAT,
                       state.radB.data() + static_cast<size_t>(t) * N,
                       state.counts.data(), state.displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        if (N > 0) {
            CUDA_CHECK(cudaMemcpy(dev.radB + static_cast<size_t>(t) * N,
                                  state.radB.data() + static_cast<size_t>(t) * N,
                                  static_cast<size_t>(N) * sizeof(val_t),
                                  cudaMemcpyHostToDevice));
        }

        if (g_rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, DeviceBuffers& dev) {
    if (g_rank == 0) printf("Computing distances via cross-correlation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int localRows = static_cast<int>(state.localRows);
    int blocks = (localRows + 7) / 8;  // 8 warps (rows) per 256-thread block
    std::vector<val_t> localOut(localRows > 0 ? state.localRows : 1, ZERO);

    if (localRows > 0) {
        distanceKernel<<<blocks, 256>>>(dev.radB, dev.distOut, N, T,
                                        static_cast<int>(state.localStart), localRows,
                                        static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(localOut.data(), dev.distOut,
                              state.localRows * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    MPI_Allgatherv(localOut.data(), localRows, MPI_FLOAT,
                   state.distances.data(), state.counts.data(), state.displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
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
    long long nonZeroKij = state.nonZeroKijGlobal;
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    g_rank = rank;

    // Determine the local rank inside this node so that each rank binds to a
    // different accelerator when several ranks share a node.
    int localRank = 0;
    int ranksOnNode = 1;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &ranksOnNode);
        MPI_Comm_free(&nodeComm);
    }

    omp_set_dynamic(0);
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        int cores = omp_get_num_procs();
        int threads = std::max(1, cores / std::max(1, ranksOnNode));
        omp_set_num_threads(threads);
    }

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        fprintf(stderr, "Error: no CUDA device available (%s)\n", cudaGetErrorString(devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int deviceId = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (ranks/node: %d)\n", nranks, ranksOnNode);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA device: %s (%d visible)\n", deviceProp.name, deviceCount);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, nranks);

    DeviceBuffers dev;
    setupDevice(state, dev);

    if (g_rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, dev);
    computeFormFactors(state, dev);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (g_rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, dev);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (g_rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, dev);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (g_rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    int rc = 0;

    if (g_rank == 0) {
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
                rc = 1;
            }
        }
    }

    freeDevice(dev);

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
