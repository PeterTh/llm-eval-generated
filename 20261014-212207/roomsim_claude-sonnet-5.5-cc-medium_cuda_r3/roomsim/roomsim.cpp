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
#include <cfloat>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

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

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
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
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
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
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Flattened octree (device representation)
// ============================================================================

struct DevOctNode {
    float c[3];       // box center
    float h[3];       // box half extent
    int child[8];     // child node indices, -1 if absent
    int triStart;     // offset into leaf triangle index list
    int triCount;     // > 0 for leaf nodes
};

static int flattenOctree(const Octree& o, std::vector<DevOctNode>& nodes, std::vector<int>& triList) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    DevOctNode n;
    n.c[0] = o.center.x; n.c[1] = o.center.y; n.c[2] = o.center.z;
    n.h[0] = o.halfExtent.x; n.h[1] = o.halfExtent.y; n.h[2] = o.halfExtent.z;
    n.triStart = 0;
    n.triCount = 0;
    if (!o.triangleIndices.empty()) {
        n.triStart = static_cast<int>(triList.size());
        n.triCount = static_cast<int>(o.triangleIndices.size());
        for (size_t idx : o.triangleIndices) triList.push_back(static_cast<int>(idx));
    }
    for (int i = 0; i < 8; ++i) {
        n.child[i] = o.children[i] ? flattenOctree(*o.children[i], nodes, triList) : -1;
    }
    nodes[id] = n;
    return id;
}

__device__ __forceinline__ bool rayIntersectsBox(const DevOctNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(nd.c[0], nd.c[1], nd.c[2]);
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const float hx = nd.h[0], hy = nd.h[1], hz = nd.h[2];

    if (fabsf(c.x) > hx + ad.x) return false;
    if (fabsf(c.y) > hy + ad.y) return false;
    if (fabsf(c.z) > hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > hy * ad.z + hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > hz * ad.x + hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > hx * ad.y + hy * ad.x + EPSILON) return false;

    return true;
}

constexpr int OCTREE_STACK_SIZE = 128;

// Check if a ray between two points is blocked by any triangle other than src/dst
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevOctNode* nodes, const int* triList, const Triangle* tris,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevOctNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (int k = 0; k < nd.triCount; ++k) {
                int idx = triList[nd.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 7; i >= 0; --i) {
                int ch = nd.child[i];
                if (ch >= 0 && rayIntersectsBox(nodes[ch], from, to)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

// ============================================================================
// Random Number Generation: parallel-friendly MT19937 stream
// ============================================================================
//
// The reference implementation draws every random number from one sequential
// std::mt19937(42) stream, with a fixed number of draws (NUM_RAYS * 4) per
// non-culled pair in row-major order. To reproduce exactly the same samples
// in parallel, a single thread block generates the raw stream into global
// memory (chunk by chunk, overlapped with consumption) and each ray reads its
// own draws at a position determined by the pair's rank.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;
// Chunk granularity: lcm(MT_N, DRAWS_PER_PAIR) words == 39 pairs
constexpr uint32_t PAIRS_PER_UNIT = 39;
constexpr uint32_t MAX_UNITS_PER_CHUNK = 6400;

__device__ __forceinline__ uint32_t mtTwist(uint32_t u, uint32_t v) {
    uint32_t y = (u & MT_UPPER) | (v & MT_LOWER);
    return (y >> 1) ^ ((v & 1u) ? MT_MATRIX_A : 0u);
}

__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// Single block of MT_LANES threads; produces numBlocks * MT_N tempered outputs
// and leaves the generator state (untempered) in 'state' for the next chunk.
// Thread t computes elements t, t+227 and t+454 of every new 624-word block:
// the later ones only depend on earlier ones of the same thread, and the state
// is ping-ponged between two shared buffers, so one barrier per block suffices.
constexpr int MT_LANES = MT_N - MT_M;  // 227

__global__ void __launch_bounds__(MT_LANES) mtGenerateKernel(uint32_t* state, uint32_t* out, int numBlocks) {
    __shared__ uint32_t buf[2][MT_N];
    const int t = threadIdx.x;
    constexpr int LAST = MT_N - 1 - 2 * MT_LANES;  // 169: last thread owning a third element
    for (int k = t; k < MT_N; k += MT_LANES) buf[0][k] = state[k];
    __syncthreads();

    // Each thread keeps its own three previous elements in registers
    uint32_t p0 = buf[0][t];
    uint32_t p1 = buf[0][t + MT_LANES];
    uint32_t p2 = (t <= LAST) ? buf[0][t + 2 * MT_LANES] : 0u;

    auto step = [&](const uint32_t* A, uint32_t* B, uint32_t* o) {
        const uint32_t x1 = A[t + 1];
        const uint32_t x397 = A[t + MT_M];
        const uint32_t x228 = A[t + MT_LANES + 1];
        uint32_t x455 = 0;
        if (t < LAST) x455 = A[t + 2 * MT_LANES + 1];
        else if (t == LAST) x455 = A[MT_M] ^ mtTwist(A[0], A[1]);  // wraps to the new element 0

        const uint32_t n0 = x397 ^ mtTwist(p0, x1);
        const uint32_t n1 = n0 ^ mtTwist(p1, x228);
        B[t] = n0;
        B[t + MT_LANES] = n1;
        o[t] = mtTemper(n0);
        o[t + MT_LANES] = mtTemper(n1);
        p0 = n0;
        p1 = n1;
        if (t <= LAST) {
            const uint32_t n2 = n1 ^ mtTwist(p2, x455);
            B[t + 2 * MT_LANES] = n2;
            o[t + 2 * MT_LANES] = mtTemper(n2);
            p2 = n2;
        }
        __syncthreads();
    };

    // numBlocks is always even
    for (int b = 0; b < numBlocks; b += 2) {
        step(buf[0], buf[1], out + static_cast<size_t>(b) * MT_N);
        step(buf[1], buf[0], out + static_cast<size_t>(b + 1) * MT_N);
    }
    for (int k = t; k < MT_N; k += MT_LANES) state[k] = buf[0][k];
}

// Equivalent of std::uniform_real_distribution<float>(0,1) on a 32-bit engine
__device__ __forceinline__ float drawToUnitFloat(uint32_t x) {
    float u = __uint2float_rn(x) / 4294967296.0f;
    return u >= 1.0f ? 0x1.fffffep-1f : u;
}

// Time delay between two triangles (in timesteps)
HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Form Factor Computation (Kij) and Tau
// ============================================================================

struct PairEntry {
    val_t w;   // min(Kij * area_j, 1), zero if no contribution
    int tau;   // time delay
};

struct KijContext {
    const Triangle* tris;
    const DevOctNode* nodes;
    const int* triList;
    const val_t* areas;
    const uint64_t* mask;         // N x WPR bitmask of non-culled pairs (i != j)
    const uint32_t* wordPrefix;   // N x WPR exclusive prefix popcount within a row
    const uint64_t* rowStart;     // N + 1 exclusive prefix of row pair counts
    PairEntry* entries;           // stored transposed: entries[j * N + i]
    unsigned long long* nonZero;
    int N;
    int WPR;
};

// Cull triangles facing the same direction (identical test to the reference)
__global__ void cullMaskKernel(const Triangle* tris, int N, int WPR, uint64_t* mask, uint32_t* wordCount) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (w >= WPR) return;
    const Triangle& triI = tris[i];
    uint64_t bits = 0;
    int jEnd = min(N, (w + 1) * 64);
    for (int j = w * 64; j < jEnd; ++j) {
        if (j == i) continue;
        if (triI.normal().dot(tris[j].normal()) > 0.99f) continue;
        bits |= 1ull << (j - w * 64);
    }
    mask[static_cast<size_t>(i) * WPR + w] = bits;
    wordCount[static_cast<size_t>(i) * WPR + w] = static_cast<uint32_t>(__popcll(bits));
}

__global__ void rowScanKernel(uint32_t* wordPrefix, uint32_t* rowTotal, int N, int WPR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t run = 0;
    for (int w = 0; w < WPR; ++w) {
        size_t idx = static_cast<size_t>(i) * WPR + w;
        uint32_t c = wordPrefix[idx];
        wordPrefix[idx] = run;
        run += c;
    }
    rowTotal[i] = run;
}

constexpr int KIJ_BLOCK = 256;

// 16 lanes cooperate on one (i, j) pair, one lane per ray.
__global__ void __launch_bounds__(KIJ_BLOCK)
kijKernel(KijContext ctx, uint64_t rank0, uint32_t nPairs, const uint32_t* __restrict__ stream) {
    const unsigned FULL = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31;
    const int r = lane & 15;
    const uint32_t g = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    const bool valid = g < nPairs;
    const int N = ctx.N;

    int i = 0, j = 0;
    if (valid && r == 0) {
        uint64_t rank = rank0 + g;
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (ctx.rowStart[mid] <= rank) lo = mid; else hi = mid - 1;
        }
        i = lo;
        uint32_t k = static_cast<uint32_t>(rank - ctx.rowStart[i]);
        const uint32_t* pre = ctx.wordPrefix + static_cast<size_t>(i) * ctx.WPR;
        int wl = 0, wh = ctx.WPR - 1;
        while (wl < wh) {
            int mid = (wl + wh + 1) >> 1;
            if (pre[mid] <= k) wl = mid; else wh = mid - 1;
        }
        uint64_t m = ctx.mask[static_cast<size_t>(i) * ctx.WPR + wl];
        uint32_t kk = k - pre[wl];
        while (kk--) m &= m - 1;
        j = wl * 64 + (__ffsll(static_cast<long long>(m)) - 1);
    }
    i = __shfl_sync(FULL, i, 0, 16);
    j = __shfl_sync(FULL, j, 0, 16);

    val_t contrib = ZERO;
    if (valid) {
        const Triangle& triI = ctx.tris[i];
        const Triangle& triJ = ctx.tris[j];
        uint4 d = *reinterpret_cast<const uint4*>(stream + static_cast<size_t>(g) * DRAWS_PER_PAIR + r * 4);

        // Random point in triangle I
        val_t u = drawToUnitFloat(d.x), v = drawToUnitFloat(d.y);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        Vec3 pI = triI.a + (triI.b - triI.a) * u + (triI.c - triI.a) * v;
        // Random point in triangle J
        u = drawToUnitFloat(d.z); v = drawToUnitFloat(d.w);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        Vec3 pJ = triJ.a + (triJ.b - triJ.a) * u + (triJ.c - triJ.a) * v;

        if (!isRayBlocked(pI, pJ, ctx.nodes, ctx.triList, ctx.tris, i, j)) {
            Vec3 vec = pJ - pI;
            val_t distSqr = vec.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosI = ZERO, cosJ = ZERO;
                val_t n1 = vec.norm();
                if (n1 > EPSILON) cosI = fmaxf(ZERO, vec.dot(triI.normal()) / n1);
                Vec3 nv = -vec;
                val_t n2 = nv.norm();
                if (n2 > EPSILON) cosJ = fmaxf(ZERO, nv.dot(triJ.normal()) / n2);
                if (cosI > ZERO && cosJ > ZERO) contrib = (cosI * cosJ) / (PI * distSqr);
            }
        }
    }

    // Sum the 16 ray contributions in the reference order
    val_t kij = ZERO;
#pragma unroll
    for (int rr = 0; rr < NUM_RAYS; ++rr) {
        kij += __shfl_sync(FULL, contrib, rr, 16);
    }
    kij *= INV_NUM_RAYS;

    bool leader = valid && r == 0;
    if (leader) {
        PairEntry e;
        e.w = fminf(kij * ctx.areas[j], ONE);
        e.tau = computeTau(ctx.tris[i], ctx.tris[j]);
        ctx.entries[static_cast<size_t>(j) * N + i] = e;
    }
    unsigned cnt = __popc(__ballot_sync(FULL, leader && kij > EPSILON));
    if (lane == 0 && cnt) atomicAdd(ctx.nonZero, static_cast<unsigned long long>(cnt));
}
// ============================================================================
// Simulation kernels
// ============================================================================

// One thread per receiving triangle i; the sum over j keeps the reference order.
__global__ void simStepKernel(const PairEntry* __restrict__ entries, const val_t* radB_prev,
                              val_t* radB, int N, int t, int sourceIdx, int timeOff, val_t rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;
#pragma unroll 8
    for (int j = 0; j < N; ++j) {
        PairEntry e = entries[static_cast<size_t>(j) * N + i];
        if (e.w > ZERO && t >= e.tau) {
            val_t radJ = radB_prev[static_cast<size_t>(t - e.tau) * N + j];
            if (radJ > ZERO) sumB += e.w * radJ;
        }
    }
    val_t radE = (i == sourceIdx && t < timeOff) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE;
}

constexpr int DIST_MAX_BLOCK = 256;

// One block per triangle; threads split the lags t. Each lag keeps the
// reference summation order; ties resolve to the smallest lag.
__global__ void distKernel(const val_t* __restrict__ radB, int N, int T, int sourceIdx,
                           val_t* distances, bool useSmem) {
    extern __shared__ val_t sm[];
    __shared__ val_t redVal[DIST_MAX_BLOCK];
    __shared__ int redT[DIST_MAX_BLOCK];

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const val_t* pB;
    const val_t* pS;
    size_t stride;
    if (useSmem) {
        val_t* sB = sm;
        val_t* sS = sm + T;
        for (int k = tid; k < T; k += blockDim.x) {
            sB[k] = radB[static_cast<size_t>(k) * N + i];
            sS[k] = radB[static_cast<size_t>(k) * N + sourceIdx];
        }
        __syncthreads();
        pB = sB; pS = sS; stride = 1;
    } else {
        pB = radB + i; pS = radB + sourceIdx; stride = N;
    }

    val_t best = ZERO;
    int bestT = 0;
    for (int t = tid; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            sum += pS[(tt - t) * stride] * pB[tt * stride];
        }
        if (sum > best) { best = sum; bestT = t; }
    }
    redVal[tid] = best;
    redT[tid] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            val_t ov = redVal[tid + s];
            int ot = redT[tid + s];
            if (ov > redVal[tid] || (ov == redVal[tid] && ot < redT[tid])) {
                redVal[tid] = ov;
                redT[tid] = ot;
            }
        }
        __syncthreads();
    }
    if (tid == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(redT[0]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;             // Reflectivity (0.0 to 1.0), uniform
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), copied back from device
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    unsigned long long nonZeroKij = 0;

    // Device resources
    Triangle* d_tris = nullptr;
    val_t* d_areas = nullptr;
    DevOctNode* d_nodes = nullptr;
    int* d_triList = nullptr;
    PairEntry* d_entries = nullptr;  // N x N (transposed), form factor weight + tau
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

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
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.reflectivity = reflectivity;
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Upload geometry and acceleration structure
    const size_t n = state.numTriangles;
    std::vector<DevOctNode> nodes;
    std::vector<int> triList;
    flattenOctree(state.octree, nodes, triList);

    CUDA_CHECK(cudaMalloc(&state.d_tris, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(DevOctNode)));
    CUDA_CHECK(cudaMalloc(&state.d_triList, triList.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_entries, n * n * sizeof(PairEntry)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, n * sizeof(val_t)));
    if (timesteps > 0) CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.d_tris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(DevOctNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triList, triList.data(), triList.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_entries, 0, n * n * sizeof(PairEntry)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, n * sizeof(val_t)));
    if (timesteps > 0) CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * n * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Tau is evaluated together with Kij inside the form factor kernel
void computeTimeDelays(SimulationState& state) {
    (void)state;
    printf("Computing time delays (Tau)...\n");
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int WPR = (N + 63) / 64;

    uint64_t* d_mask;
    uint32_t* d_wordPrefix;
    uint32_t* d_rowTotal;
    uint64_t* d_rowStart;
    unsigned long long* d_nonZero;
    CUDA_CHECK(cudaMalloc(&d_mask, static_cast<size_t>(N) * WPR * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_wordPrefix, static_cast<size_t>(N) * WPR * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rowTotal, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rowStart, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_nonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_nonZero, 0, sizeof(unsigned long long)));

    // Which pairs consume random numbers, and their rank in row-major order
    {
        dim3 grid((WPR + 63) / 64, N);
        cullMaskKernel<<<grid, 64>>>(state.d_tris, N, WPR, d_mask, d_wordPrefix);
        CUDA_CHECK(cudaGetLastError());
        rowScanKernel<<<(N + 127) / 128, 128>>>(d_wordPrefix, d_rowTotal, N, WPR);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<uint32_t> rowTotal(N);
    CUDA_CHECK(cudaMemcpy(rowTotal.data(), d_rowTotal, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> rowStart(N + 1, 0);
    for (int i = 0; i < N; ++i) rowStart[i + 1] = rowStart[i] + rowTotal[i];
    CUDA_CHECK(cudaMemcpy(d_rowStart, rowStart.data(), (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    const uint64_t totalPairs = rowStart[N];

    if (totalPairs > 0) {
        // Initial MT19937 state for seed 42
        std::vector<uint32_t> mt(MT_N);
        mt[0] = 42u;
        for (int k = 1; k < MT_N; ++k) mt[k] = 1812433253u * (mt[k - 1] ^ (mt[k - 1] >> 30)) + static_cast<uint32_t>(k);
        uint32_t* d_mtState;
        CUDA_CHECK(cudaMalloc(&d_mtState, MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_mtState, mt.data(), MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        const uint64_t unitsTotal = (totalPairs + PAIRS_PER_UNIT - 1) / PAIRS_PER_UNIT;
        const uint32_t unitsPerChunk = static_cast<uint32_t>(std::min<uint64_t>(unitsTotal, MAX_UNITS_PER_CHUNK));
        const uint64_t pairsPerChunk = static_cast<uint64_t>(unitsPerChunk) * PAIRS_PER_UNIT;
        const size_t chunkWords = static_cast<size_t>(pairsPerChunk) * DRAWS_PER_PAIR;
        const uint64_t numChunks = (totalPairs + pairsPerChunk - 1) / pairsPerChunk;

        uint32_t* d_stream[2];
        cudaEvent_t genDone[2], useDone[2];
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&d_stream[b], chunkWords * sizeof(uint32_t)));
            CUDA_CHECK(cudaEventCreateWithFlags(&genDone[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&useDone[b], cudaEventDisableTiming));
        }
        cudaStream_t genStream, compStream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&genStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compStream, cudaStreamNonBlocking));
        // Make sure the setup work on the default stream is complete
        CUDA_CHECK(cudaDeviceSynchronize());

        KijContext ctx;
        ctx.tris = state.d_tris;
        ctx.nodes = state.d_nodes;
        ctx.triList = state.d_triList;
        ctx.areas = state.d_areas;
        ctx.mask = d_mask;
        ctx.wordPrefix = d_wordPrefix;
        ctx.rowStart = d_rowStart;
        ctx.entries = state.d_entries;
        ctx.nonZero = d_nonZero;
        ctx.N = N;
        ctx.WPR = WPR;

        for (uint64_t c = 0; c < numChunks; ++c) {
            const int b = static_cast<int>(c & 1);
            const uint64_t rank0 = c * pairsPerChunk;
            const uint32_t nPairs = static_cast<uint32_t>(std::min<uint64_t>(pairsPerChunk, totalPairs - rank0));

            if (c >= 2) CUDA_CHECK(cudaStreamWaitEvent(genStream, useDone[b], 0));
            // Only generate the words actually needed (whole MT blocks)
            const uint32_t units = (nPairs + PAIRS_PER_UNIT - 1) / PAIRS_PER_UNIT;
            mtGenerateKernel<<<1, MT_LANES, 0, genStream>>>(d_mtState, d_stream[b], static_cast<int>(units * 4));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(genDone[b], genStream));

            CUDA_CHECK(cudaStreamWaitEvent(compStream, genDone[b], 0));
            const uint64_t threads = static_cast<uint64_t>(nPairs) * 16;
            const unsigned blocks = static_cast<unsigned>((threads + KIJ_BLOCK - 1) / KIJ_BLOCK);
            kijKernel<<<blocks, KIJ_BLOCK, 0, compStream>>>(ctx, rank0, nPairs, d_stream[b]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(useDone[b], compStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(compStream));
        CUDA_CHECK(cudaStreamSynchronize(genStream));

        for (int b = 0; b < 2; ++b) {
            cudaFree(d_stream[b]);
            cudaEventDestroy(genDone[b]);
            cudaEventDestroy(useDone[b]);
        }
        cudaStreamDestroy(genStream);
        cudaStreamDestroy(compStream);
        cudaFree(d_mtState);
    }

    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, d_nonZero, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_mask);
    cudaFree(d_wordPrefix);
    cudaFree(d_rowTotal);
    cudaFree(d_rowStart);
    cudaFree(d_nonZero);

    for (size_t i = 99; i < state.numTriangles; i += 100) {
        printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
    }
    if (state.numTriangles % 100 != 0) {
        printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = T / 2;

    // Time steps depend on earlier ones (tau >= 1), so they run in order
    for (int t = 0; t < T; ++t) {
        simStepKernel<<<(N + 127) / 128, 128>>>(state.d_entries, state.d_radB, state.d_radB, N, t,
                                                static_cast<int>(state.sourceIndex), timeOff,
                                                state.reflectivity);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int t = 0; t < T; ++t) {
        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    if (T > 0) {
        int block = std::min(DIST_MAX_BLOCK, std::max(32, (T + 31) / 32 * 32));
        // block must be a power of two for the tree reduction
        int pow2 = 32;
        while (pow2 < block) pow2 <<= 1;
        block = pow2;

        size_t smemBytes = 2 * static_cast<size_t>(T) * sizeof(val_t);
        int dev = 0, maxOptin = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        bool useSmem = smemBytes + 2 * DIST_MAX_BLOCK * sizeof(int) <= static_cast<size_t>(maxOptin);
        if (useSmem) {
            CUDA_CHECK(cudaFuncSetAttribute(distKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smemBytes)));
        } else {
            smemBytes = 0;
        }
        distKernel<<<N, block, smemBytes>>>(state.d_radB, N, T, static_cast<int>(state.sourceIndex),
                                            state.d_distances, useSmem);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, N * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (T > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, state.radB.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

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
            return 1;
        }
    }

    return 0;
}
