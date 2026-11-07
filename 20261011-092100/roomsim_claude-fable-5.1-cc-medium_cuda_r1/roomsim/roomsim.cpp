/**
 * Room Response Simulation Benchmark (CUDA version)
 *
 * GPU-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization notes
 * ---------------------
 * All three heavy phases run on the GPU:
 *
 *  - Form factors: one GPU thread per (triangle pair, ray). Visibility rays are
 *    traced through a flattened copy of the CPU-built octree. The Monte Carlo
 *    sampling reproduces the exact random stream of the sequential program:
 *    the sequential code draws from a single std::mt19937 (seed 42) in row-major
 *    pair order, consuming exactly 64 numbers for every pair that is not culled.
 *    The host advances the MT19937 state (the only inherently sequential part,
 *    ~1 ns per 624 words thanks to vectorization) and emits a 624-word
 *    checkpoint every K twists; GPU blocks expand each checkpoint into the
 *    tempered output words, and the ray threads pick up their numbers from
 *    the global stream position of their pair. Work is pipelined in batches
 *    over two CUDA streams so host RNG generation, transfers and GPU ray
 *    tracing overlap.
 *  - Wave propagation: one kernel per timestep, one thread per receiving
 *    triangle, summing contributions over emitters in the original order
 *    (Kij stored transposed for coalesced access, Tau recomputed on the fly).
 *  - Distances: one block per triangle, one thread per lag, with a
 *    first-maximum reduction equivalent to the sequential argmax.
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
#include <vector>

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

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                           \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

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

    HD bool operator==(const Vec3& o) const {
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
// Octree for Spatial Acceleration (built on the host, flattened for the GPU)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

// Flattened octree for the GPU traversal. A node stores its leaf triangle
// range and its child indices; the children's bounding boxes are stored
// contiguously per parent (8 entries) so that the box tests of one node read
// a single memory region. Leaf children carry LEAF_BIT in their index.
constexpr int LEAF_BIT = 0x40000000;

struct __align__(16) GpuNode {
    int triStart;           // first entry in the leaf triangle list (leaf only)
    int triCount;           // number of triangles (0 for internal nodes)
    int child[8];           // child node indices (| LEAF_BIT for leaves), -1 if absent
    int pad[2];
};

struct GpuOctree {
    std::vector<GpuNode> nodes;
    std::vector<float4> childBox0;   // per node, 8 x (cx, cy, cz, hx)
    std::vector<float2> childBox1;   // per node, 8 x (hy, hz)
    std::vector<float4> leafTri;     // per leaf entry: a (w = triangle index bits), b, c
    int rootEncoded = 0;
};

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

    // Flatten the tree into arrays (depth-first order) for GPU traversal.
    void flatten(GpuOctree& out) const {
        out.nodes.clear();
        out.childBox0.clear();
        out.childBox1.clear();
        out.leafTri.clear();
        int root = flattenNode(out);
        out.rootEncoded = root;
        if (!triangleIndices.empty()) out.rootEncoded |= LEAF_BIT;
    }

private:
    int flattenNode(GpuOctree& out) const {
        int myIndex = static_cast<int>(out.nodes.size());
        out.nodes.emplace_back();
        for (int i = 0; i < 8; ++i) {
            out.childBox0.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            out.childBox1.push_back(make_float2(0.0f, 0.0f));
        }
        GpuNode n;
        n.triStart = static_cast<int>(out.leafTri.size() / 3);
        n.triCount = static_cast<int>(triangleIndices.size());
        n.pad[0] = n.pad[1] = 0;
        for (size_t idx : triangleIndices) {
            const Triangle& t = (*allTriangles)[idx];
            int id = static_cast<int>(idx);
            float idBits;
            memcpy(&idBits, &id, sizeof(float));
            out.leafTri.push_back(make_float4(t.a.x, t.a.y, t.a.z, idBits));
            out.leafTri.push_back(make_float4(t.b.x, t.b.y, t.b.z, 0.0f));
            out.leafTri.push_back(make_float4(t.c.x, t.c.y, t.c.z, 0.0f));
        }
        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                const Octree& c = *children[i];
                int ci = c.flattenNode(out);
                n.child[i] = c.triangleIndices.empty() ? ci : (ci | LEAF_BIT);
                out.childBox0[static_cast<size_t>(myIndex) * 8 + i] =
                    make_float4(c.center.x, c.center.y, c.center.z, c.halfExtent.x);
                out.childBox1[static_cast<size_t>(myIndex) * 8 + i] =
                    make_float2(c.halfExtent.y, c.halfExtent.z);
            } else {
                n.child[i] = -1;
            }
        }
        out.nodes[myIndex] = n;
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
//
// The sequential program draws all form factor samples from one
// std::mt19937(42) through std::uniform_real_distribution<float>(0,1).
// To reproduce that stream bit-exactly on the GPU, the host advances the
// MT19937 state array ("twists") and hands out 624-word checkpoints; the GPU
// regenerates the tempered outputs in between. The conversion to float
// matches libstdc++'s generate_canonical<float,24>: float(x) / 2^32, with the
// result clamped below 1.0f.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;

constexpr int MT_TWISTS_PER_CHECKPOINT = 64;                       // K
constexpr int MT_WORDS_PER_CHECKPOINT = MT_TWISTS_PER_CHECKPOINT * MT_N;
constexpr int RANDS_PER_PAIR = NUM_RAYS * 4;                       // 64 draws per sampled pair

class MtCheckpointGenerator {
    uint32_t mt[MT_N];
    uint64_t twists = 0;
    uint32_t lastCp[MT_N];
    int64_t lastCpIndex = -1;

    void twist() {
        uint32_t* m = mt;
        for (int i = 0; i < MT_N - MT_M; ++i) {
            uint32_t y = (m[i] & MT_UPPER) | (m[i + 1] & MT_LOWER);
            m[i] = m[i + MT_M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        }
        for (int i = MT_N - MT_M; i < MT_N - 1; ++i) {
            uint32_t y = (m[i] & MT_UPPER) | (m[i + 1] & MT_LOWER);
            m[i] = m[i + MT_M - MT_N] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        }
        uint32_t y = (m[MT_N - 1] & MT_UPPER) | (m[0] & MT_LOWER);
        m[MT_N - 1] = m[MT_M - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        ++twists;
    }

public:
    explicit MtCheckpointGenerator(uint32_t seed) {
        mt[0] = seed;
        for (int i = 1; i < MT_N; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }

    // Checkpoint c holds the state after c*K twists, i.e. the 624 raw words
    // preceding output index c*K*624. Checkpoints must be requested in
    // non-decreasing order (the last one may be requested again).
    void getCheckpoint(uint64_t c, uint32_t* dst) {
        if (static_cast<int64_t>(c) == lastCpIndex) {
            memcpy(dst, lastCp, sizeof(lastCp));
            return;
        }
        uint64_t target = c * MT_TWISTS_PER_CHECKPOINT;
        if (target < twists) {
            fprintf(stderr, "MT checkpoint requested out of order\n");
            exit(1);
        }
        while (twists < target) twist();
        memcpy(dst, mt, sizeof(mt));
        memcpy(lastCp, mt, sizeof(mt));
        lastCpIndex = static_cast<int64_t>(c);
    }
};

__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0,1) applied to a 32-bit mt19937 output
__device__ __forceinline__ float mtCanonical(uint32_t x) {
    float f = __uint2float_rn(x) * 2.3283064365386962890625e-10f;  // / 2^32 (exact scaling)
    if (f >= 1.0f) f = 0.99999994f;                                 // nextafter(1, 0)
    return f;
}

// Expand checkpoint blockIdx.x into K*624 tempered output words.
// Output word n (relative to the checkpoint) is r_n with
// r_n = r_{n-227} ^ twist(r_{n-624}, r_{n-623}); the 227 words of a step only
// depend on words before the step, so they are computed in parallel.
__global__ void __launch_bounds__(256)
kernelExpandMt(const uint32_t* __restrict__ checkpoints, uint32_t* __restrict__ out) {
    __shared__ uint32_t ring[1024];
    const uint32_t* cp = checkpoints + static_cast<size_t>(blockIdx.x) * MT_N;
    uint32_t* o = out + static_cast<size_t>(blockIdx.x) * MT_WORDS_PER_CHECKPOINT;

    for (int i = threadIdx.x; i < MT_N; i += blockDim.x) {
        ring[(i - MT_N + 2048) & 1023] = cp[i];
    }
    __syncthreads();

    constexpr int STEP = MT_N - MT_M;  // 227
    for (int base = 0; base < MT_WORDS_PER_CHECKPOINT; base += STEP) {
        int p = base + static_cast<int>(threadIdx.x);
        if (threadIdx.x < STEP && p < MT_WORDS_PER_CHECKPOINT) {
            uint32_t a = ring[(p - MT_N + 2048) & 1023];
            uint32_t b = ring[(p - MT_N + 1 + 2048) & 1023];
            uint32_t c = ring[(p - STEP + 2048) & 1023];
            uint32_t y = (a & MT_UPPER) | (b & MT_LOWER);
            uint32_t r = c ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
            ring[p & 1023] = r;
            o[p] = mtTemper(r);
        }
        __syncthreads();
    }
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ Vec3 randomPointInTriangle(const Vec3& a, const Vec3& b, const Vec3& c,
                                                      val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                              const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 3.402823466e+38f;  // numeric_limits<float>::max()

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e+38f;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, GPU)
// ============================================================================

__device__ __forceinline__ Vec3 loadVec3(const float4* p, int i) {
    float4 v = p[i];
    return Vec3(v.x, v.y, v.z);
}

// Check if a ray segment intersects a node's bounding box (same test as the CPU
// octree). d = (p2 - p1) / 2 and ad = |d| are shared by all tests of one ray.
__device__ __forceinline__ bool rayIntersectsBox(const float4& box0, const float2& box1,
                                                 const Vec3& p1, const Vec3& d, const Vec3& ad) {
    Vec3 c = p1 + d - Vec3(box0.x, box0.y, box0.z);
    const val_t hx = box0.w, hy = box1.x, hz = box1.y;

    if (fabsf(c.x) > hx + ad.x) return false;
    if (fabsf(c.y) > hy + ad.y) return false;
    if (fabsf(c.z) > hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > hy * ad.z + hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > hz * ad.x + hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > hx * ad.y + hy * ad.x + EPSILON) return false;

    return true;
}

struct GpuScene {
    const float4* triA;
    const float4* triB;
    const float4* triC;
    const float4* triN;
    const GpuNode* nodes;
    const float4* childBox0;
    const float2* childBox1;
    const float4* leafTri;
    int rootEncoded;
};

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const GpuScene& scene,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Segment-vs-box quantities shared by all box tests of this ray
    Vec3 hd = (to - from) * 0.5f;
    Vec3 ahd = {fabsf(hd.x), fabsf(hd.y), fabsf(hd.z)};

    // Depth-first traversal with an explicit stack. The result (any hit) does
    // not depend on the visiting order. Internal nodes and leaves are handled
    // in separate loops to reduce divergence between the lanes of a warp.
    int stack[64];
    int sp = 0;
    stack[sp++] = scene.rootEncoded;  // root is never box-tested, as in the sequential code

    while (sp > 0) {
        // Internal nodes until a leaf is on top of the stack
        while (sp > 0 && !(stack[sp - 1] & LEAF_BIT)) {
            const int ni = stack[--sp];
            const int4* np = reinterpret_cast<const int4*>(scene.nodes + ni);
            const int4 m0 = __ldg(np);        // triStart, triCount, child[0], child[1]
            const int4 m1 = __ldg(np + 1);    // child[2..5]
            const int4 m2 = __ldg(np + 2);    // child[6], child[7], pad
            const int ch[8] = {m0.z, m0.w, m1.x, m1.y, m1.z, m1.w, m2.x, m2.y};
            const float4* cb0 = scene.childBox0 + static_cast<size_t>(ni) * 8;
            const float2* cb1 = scene.childBox1 + static_cast<size_t>(ni) * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c = ch[i];
                if (c >= 0 && rayIntersectsBox(__ldg(cb0 + i), __ldg(cb1 + i), from, hd, ahd)) {
                    stack[sp++] = c;
                }
            }
        }
        // Leaves: test triangles directly
        while (sp > 0 && (stack[sp - 1] & LEAF_BIT)) {
            const int ni = stack[--sp] & ~LEAF_BIT;
            const int4 m0 = __ldg(reinterpret_cast<const int4*>(scene.nodes + ni));
            const float4* lt = scene.leafTri + static_cast<size_t>(m0.x) * 3;
            for (int k = 0; k < m0.y; ++k) {
                const float4 a = __ldg(lt + 3 * k);
                const int idx = __float_as_int(a.w);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const float4 b = __ldg(lt + 3 * k + 1);
                const float4 c = __ldg(lt + 3 * k + 2);
                val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(a.x, a.y, a.z),
                                                  Vec3(b.x, b.y, b.z), Vec3(c.x, c.y, c.z));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - GPU kernels
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// A pair (i, j) consumes random numbers iff it is not skipped by the sequential code.
__device__ __forceinline__ bool pairSampled(int i, int j, const float4* triN) {
    if (i == j) return false;
    float4 ni = triN[i];
    float4 nj = triN[j];
    val_t d = ni.x * nj.x + ni.y * nj.y + ni.z * nj.z;
    return !(d > 0.99f);
}

constexpr int SEG_LEN = 256;  // columns per (row, segment) unit

// Count sampled pairs per (row, segment) unit. One warp per unit.
__global__ void kernelCountSegments(int N, int unitsPerRow, const float4* __restrict__ triN,
                                    uint32_t* __restrict__ segCount) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    long long totalUnits = static_cast<long long>(N) * unitsPerRow;
    if (warp >= totalUnits) return;
    int i = warp / unitsPerRow;
    int s = warp % unitsPerRow;
    int jBegin = s * SEG_LEN;
    int jEnd = min(jBegin + SEG_LEN, N);
    uint32_t count = 0;
    for (int j0 = jBegin; j0 < jEnd; j0 += 32) {
        int j = j0 + lane;
        bool f = (j < jEnd) && pairSampled(i, j, triN);
        unsigned m = __ballot_sync(0xffffffffu, f);
        count += __popc(m);
    }
    if (lane == 0) segCount[warp] = count;
}

// Write the compact list of sampled pairs for units [u0, u1) of a batch,
// positioned by their global sample index minus k0. One warp per unit.
__global__ void kernelBuildPairList(int N, int unitsPerRow, const float4* __restrict__ triN,
                                    const unsigned long long* __restrict__ segStart,
                                    long long u0, long long u1, unsigned long long k0,
                                    uint2* __restrict__ list) {
    long long unit = u0 + ((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (unit >= u1) return;
    int i = static_cast<int>(unit / unitsPerRow);
    int s = static_cast<int>(unit % unitsPerRow);
    int jBegin = s * SEG_LEN;
    int jEnd = min(jBegin + SEG_LEN, N);
    unsigned long long pos = segStart[unit] - k0;
    unsigned laneMaskLt = (1u << lane) - 1u;
    for (int j0 = jBegin; j0 < jEnd; j0 += 32) {
        int j = j0 + lane;
        bool f = (j < jEnd) && pairSampled(i, j, triN);
        unsigned m = __ballot_sync(0xffffffffu, f);
        if (f) {
            list[pos + __popc(m & laneMaskLt)] = make_uint2(static_cast<unsigned>(i),
                                                            static_cast<unsigned>(j));
        }
        pos += __popc(m);
    }
}

// Form factor kernel: 16 consecutive threads handle the 16 rays of one pair.
__global__ void __launch_bounds__(256)
kernelFormFactors(const uint2* __restrict__ list, int nPairs,
                  const uint32_t* __restrict__ rands, unsigned long long randsBase,
                  unsigned long long k0, GpuScene scene, int N,
                  float* __restrict__ kijT) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int pairLocal = gtid >> 4;
    int ray = threadIdx.x & 15;
    if (pairLocal >= nPairs) return;

    uint2 ij = list[pairLocal];
    int i = static_cast<int>(ij.x);
    int j = static_cast<int>(ij.y);

    // Random draws for this ray: the sequential code draws u,v for pI then u,v for pJ
    unsigned long long w = (k0 + pairLocal) * RANDS_PER_PAIR + ray * 4 - randsBase;
    uint4 words = *reinterpret_cast<const uint4*>(rands + w);
    val_t uI = mtCanonical(words.x);
    val_t vI = mtCanonical(words.y);
    val_t uJ = mtCanonical(words.z);
    val_t vJ = mtCanonical(words.w);

    Vec3 aI = loadVec3(scene.triA, i), bI = loadVec3(scene.triB, i), cI = loadVec3(scene.triC, i);
    Vec3 aJ = loadVec3(scene.triA, j), bJ = loadVec3(scene.triB, j), cJ = loadVec3(scene.triC, j);
    Vec3 nI = loadVec3(scene.triN, i);
    Vec3 nJ = loadVec3(scene.triN, j);

    Vec3 pI = randomPointInTriangle(aI, bI, cI, uI, vI);
    Vec3 pJ = randomPointInTriangle(aJ, bJ, cJ, uJ, vJ);

    val_t contrib = ZERO;
    if (!isRayBlocked(pI, pJ, scene, i, j)) {
        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (!(distSqr < EPSILON)) {
            val_t cosPhiI = cosPhi(v, nI);
            val_t cosPhiJ = cosPhi(-v, nJ);
            if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }
    }

    // Sum the 16 ray contributions in ray order (same order as the sequential loop).
    unsigned mask = 0xffffu << (threadIdx.x & 16);
    val_t kij = __shfl_sync(mask, contrib, 0, 16);
    #pragma unroll
    for (int r = 1; r < NUM_RAYS; ++r) {
        kij += __shfl_sync(mask, contrib, r, 16);
    }

    if (ray == 0) {
        kijT[static_cast<size_t>(j) * N + i] = kij * INV_NUM_RAYS;
    }
}

// ============================================================================
// Simulation Kernel (Wave Propagation)
// ============================================================================
//
// Each block owns SIM_ROWS receiving triangles. For every chunk of SIM_CHUNK
// emitters the whole block computes the (independent) pair contributions into
// shared memory; then one thread per row accumulates them in emitter order,
// which keeps the floating-point summation identical to the sequential loop.
// Contributions the sequential code skips are stored as exact zeros.

constexpr int SIM_ROWS = 16;
constexpr int SIM_CHUNK = 256;
constexpr int SIM_THREADS = 256;
constexpr int SIM_PITCH = SIM_CHUNK + 1;   // padded to avoid shared memory bank conflicts

__global__ void __launch_bounds__(SIM_THREADS)
kernelSimStep(int t, int N, const float* __restrict__ kijT, const float4* __restrict__ centers,
              const float* __restrict__ areas, const float* __restrict__ rho,
              const float* __restrict__ radE, float* __restrict__ radB) {
    __shared__ float contrib[SIM_ROWS * SIM_PITCH];

    const int i0 = blockIdx.x * SIM_ROWS;
    const int r = threadIdx.x % SIM_ROWS;   // row of the group handled by this thread
    const int i = i0 + r;
    const bool rowValid = i < N;
    const float4 ci = rowValid ? centers[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    val_t sumB = ZERO;

    for (int j0 = 0; j0 < N; j0 += SIM_CHUNK) {
        // Phase A: pair contributions for rows [i0, i0+SIM_ROWS) x emitters [j0, j0+SIM_CHUNK)
        for (int idx = threadIdx.x; idx < SIM_ROWS * SIM_CHUNK; idx += SIM_THREADS) {
            const int jj = idx / SIM_ROWS;       // idx % SIM_ROWS == r
            const int j = j0 + jj;
            val_t c = ZERO;
            if (rowValid && j < N && j != i) {
                // Time delay Tau(i, j) = ceil(|c_i - c_j| / WAVE_SPEED)
                float4 cj = centers[j];
                Vec3 d(ci.x - cj.x, ci.y - cj.y, ci.z - cj.z);
                int tauij = static_cast<int>(ceilf(d.norm() * INV_WAVE_SPEED));
                int srcTime = t - tauij;
                // Skip if wave hasn't yet propagated from j to i
                if (srcTime >= 0) {
                    val_t kij = kijT[static_cast<size_t>(j) * N + i];
                    if (kij > ZERO) {
                        // Radiosity from source triangle at time when emission occurred
                        val_t radJ = radB[static_cast<size_t>(srcTime) * N + j];
                        if (radJ > ZERO) {
                            // form factor * area * source radiosity
                            // (separate multiply and add, as in the sequential build)
                            c = __fmul_rn(fminf(kij * areas[j], ONE), radJ);
                        }
                    }
                }
            }
            contrib[r * SIM_PITCH + jj] = c;
        }
        __syncthreads();

        // Phase B: ordered accumulation, one thread per row
        if (threadIdx.x < SIM_ROWS) {
            const int jEnd = min(SIM_CHUNK, N - j0);
            const float* row = contrib + threadIdx.x * SIM_PITCH;
            #pragma unroll 8
            for (int jj = 0; jj < jEnd; ++jj) sumB += row[jj];
        }
        __syncthreads();
    }

    // Update radiosity: reflection + emission
    if (threadIdx.x < SIM_ROWS && rowValid) {
        radB[static_cast<size_t>(t) * N + i] =
            fmaf(rho[i], sumB, radE[static_cast<size_t>(t) * N + i]);
    }
}

// ============================================================================
// Distance Kernel (Cross-Correlation): one block per triangle, thread per lag
// ============================================================================

__global__ void kernelDistances(int N, int T, int sourceIndex, const float* __restrict__ radB,
                                float* __restrict__ distances) {
    extern __shared__ float sh[];
    float* sB = sh;         // radB[:, i]
    float* sS = sh + T;     // radB[:, source]
    __shared__ float redVal[32];
    __shared__ int redT[32];

    int i = blockIdx.x;
    for (int tt = threadIdx.x; tt < T; tt += blockDim.x) {
        sB[tt] = radB[static_cast<size_t>(tt) * N + i];
        sS[tt] = radB[static_cast<size_t>(tt) * N + sourceIndex];
    }
    __syncthreads();

    // Sequential semantics: update only on strictly larger correlation, so the
    // result is the smallest lag reaching the maximum (0 if no positive value).
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            sum = fmaf(sS[tt - t], sB[tt], sum);
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    // Reduce (maxCorr, bestT): larger value wins, ties -> smaller lag
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        val_t ov = __shfl_down_sync(0xffffffffu, maxCorr, off);
        int ot = __shfl_down_sync(0xffffffffu, bestT, off);
        if (ov > maxCorr || (ov == maxCorr && ot < bestT)) {
            maxCorr = ov;
            bestT = ot;
        }
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        redVal[warp] = maxCorr;
        redT[warp] = bestT;
    }
    __syncthreads();
    if (warp == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        maxCorr = (lane < numWarps) ? redVal[lane] : ZERO;
        bestT = (lane < numWarps) ? redT[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val_t ov = __shfl_down_sync(0xffffffffu, maxCorr, off);
            int ot = __shfl_down_sync(0xffffffffu, bestT, off);
            if (ov > maxCorr || (ov == maxCorr && ot < bestT)) {
                maxCorr = ov;
                bestT = ot;
            }
        }
        if (lane == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct GpuState {
    float4* triA = nullptr;
    float4* triB = nullptr;
    float4* triC = nullptr;
    float4* triN = nullptr;
    float4* centers = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    GpuNode* nodes = nullptr;
    float4* childBox0 = nullptr;
    float2* childBox1 = nullptr;
    float4* leafTri = nullptr;
    int rootEncoded = 0;
    float* kijT = nullptr;      // Form factors, transposed: kijT[j * N + i] = Kij
    float* radE = nullptr;      // T x N
    float* radB = nullptr;      // T x N
    float* distances = nullptr;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix; filled from GPU for validation)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    GpuState gpu;                   // Device-side copies

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

    const size_t N = state.numTriangles;

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    // Initialize matrices
    state.radE.resize(timesteps * N, ZERO);
    state.radB.resize(timesteps * N, ZERO);
    state.distances.resize(N, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---- Upload scene to the GPU ----
    printf("Uploading scene to GPU...\n");
    GpuState& g = state.gpu;

    std::vector<float4> hA(N), hB(N), hC(N), hNrm(N), hCen(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 c = t.center();
        hA[i] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        hB[i] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        hC[i] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        hNrm[i] = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
        hCen[i] = make_float4(c.x, c.y, c.z, 0.0f);
    }

    GpuOctree oct;
    state.octree.flatten(oct);
    if (oct.leafTri.empty()) oct.leafTri.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    g.rootEncoded = oct.rootEncoded;

    CUDA_CHECK(cudaMalloc(&g.triA, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.triB, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.triC, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.triN, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.centers, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.nodes, oct.nodes.size() * sizeof(GpuNode)));
    CUDA_CHECK(cudaMalloc(&g.childBox0, oct.childBox0.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.childBox1, oct.childBox1.size() * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&g.leafTri, oct.leafTri.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.kijT, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.radE, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.radB, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.distances, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(g.triA, hA.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.triB, hB.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.triC, hC.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.triN, hNrm.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.centers, hCen.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.nodes, oct.nodes.data(), oct.nodes.size() * sizeof(GpuNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.childBox0, oct.childBox0.data(), oct.childBox0.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.childBox1, oct.childBox1.data(), oct.childBox1.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.leafTri, oct.leafTri.data(), oct.leafTri.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.radE, state.radE.data(), timesteps * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.kijT, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.radB, 0, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.distances, 0, N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeGpu(GpuState& g) {
    cudaFree(g.triA); cudaFree(g.triB); cudaFree(g.triC); cudaFree(g.triN);
    cudaFree(g.centers); cudaFree(g.areas); cudaFree(g.rho);
    cudaFree(g.nodes); cudaFree(g.childBox0); cudaFree(g.childBox1); cudaFree(g.leafTri);
    cudaFree(g.kijT);
    cudaFree(g.radE); cudaFree(g.radB); cudaFree(g.distances);
    g = GpuState();
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);
    GpuState& g = state.gpu;
    GpuScene scene{g.triA, g.triB, g.triC, g.triN, g.nodes, g.childBox0, g.childBox1,
                   g.leafTri, g.rootEncoded};

    // ---- 1. Count sampled pairs per (row, segment) unit and prefix-sum them ----
    const int unitsPerRow = (N + SEG_LEN - 1) / SEG_LEN;
    const long long numUnits = static_cast<long long>(N) * unitsPerRow;

    uint32_t* dSegCount = nullptr;
    unsigned long long* dSegStart = nullptr;
    CUDA_CHECK(cudaMalloc(&dSegCount, numUnits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dSegStart, (numUnits + 1) * sizeof(unsigned long long)));
    {
        int threads = 256;
        long long blocks = (numUnits * 32 + threads - 1) / threads;
        kernelCountSegments<<<static_cast<unsigned>(blocks), threads>>>(N, unitsPerRow, g.triN, dSegCount);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<uint32_t> segCount(numUnits);
    CUDA_CHECK(cudaMemcpy(segCount.data(), dSegCount, numUnits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> segStart(numUnits + 1);
    segStart[0] = 0;
    for (long long u = 0; u < numUnits; ++u) segStart[u + 1] = segStart[u] + segCount[u];
    CUDA_CHECK(cudaMemcpy(dSegStart, segStart.data(), (numUnits + 1) * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    const unsigned long long totalPairs = segStart[numUnits];

    // ---- 2. Batched, double-buffered pipeline ----
    // Batch buffers are sized for up to 2^20 pairs (64 M random words), or less for small problems.
    const unsigned long long maxBatchPairs =
        std::max(1ull, std::min(1ull << 20, (totalPairs + 1) / 2 + 1));
    const size_t scratchWords = (static_cast<size_t>(maxBatchPairs) * RANDS_PER_PAIR / MT_WORDS_PER_CHECKPOINT + 2)
                                * MT_WORDS_PER_CHECKPOINT;
    const size_t maxCp = scratchWords / MT_WORDS_PER_CHECKPOINT;

    cudaStream_t streams[2];
    cudaEvent_t h2dDone[2];
    uint32_t* hCp[2];
    uint32_t* dCp[2];
    uint32_t* dScratch[2];
    uint2* dList[2];
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost(&hCp[s], maxCp * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dCp[s], maxCp * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dScratch[s], scratchWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dList[s], maxBatchPairs * sizeof(uint2)));
    }

    MtCheckpointGenerator rng(42);

    long long u = 0;
    int batch = 0;
    long long rowsReported = 0;
    while (u < numUnits) {
        long long u0 = u;
        unsigned long long k0 = segStart[u0];
        long long u1 = u0 + 1;
        while (u1 < numUnits && segStart[u1 + 1] - k0 <= maxBatchPairs) ++u1;
        unsigned long long k1 = segStart[u1];
        u = u1;
        if (k1 == k0) continue;

        const int slot = batch & 1;
        cudaStream_t st = streams[slot];

        // Random stream words [k0*64, k1*64) are needed; expand whole checkpoints covering them.
        unsigned long long w0 = k0 * RANDS_PER_PAIR;
        unsigned long long w1 = k1 * RANDS_PER_PAIR;
        unsigned long long c0 = w0 / MT_WORDS_PER_CHECKPOINT;
        unsigned long long c1 = (w1 + MT_WORDS_PER_CHECKPOINT - 1) / MT_WORDS_PER_CHECKPOINT;
        size_t nCp = static_cast<size_t>(c1 - c0);

        // Host side: generate checkpoints (overlaps with GPU work of the previous batch)
        CUDA_CHECK(cudaEventSynchronize(h2dDone[slot]));
        for (unsigned long long c = c0; c < c1; ++c) {
            rng.getCheckpoint(c, hCp[slot] + static_cast<size_t>(c - c0) * MT_N);
        }

        CUDA_CHECK(cudaMemcpyAsync(dCp[slot], hCp[slot], nCp * MT_N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaEventRecord(h2dDone[slot], st));

        kernelExpandMt<<<static_cast<unsigned>(nCp), 256, 0, st>>>(dCp[slot], dScratch[slot]);
        CUDA_CHECK(cudaGetLastError());

        {
            long long units = u1 - u0;
            int threads = 256;
            long long blocks = (units * 32 + threads - 1) / threads;
            kernelBuildPairList<<<static_cast<unsigned>(blocks), threads, 0, st>>>(
                N, unitsPerRow, g.triN, dSegStart, u0, u1, k0, dList[slot]);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int nPairs = static_cast<int>(k1 - k0);
            int threads = 256;
            long long blocks = (static_cast<long long>(nPairs) * NUM_RAYS + threads - 1) / threads;
            kernelFormFactors<<<static_cast<unsigned>(blocks), threads, 0, st>>>(
                dList[slot], nPairs, dScratch[slot], c0 * MT_WORDS_PER_CHECKPOINT, k0,
                scene, N, g.kijT);
            CUDA_CHECK(cudaGetLastError());
        }

        ++batch;
        long long rowsDone = u1 / unitsPerRow;
        if (rowsDone / 100 > rowsReported / 100 || rowsDone == N) {
            printf("  Progress: %lld/%d triangles\n", rowsDone, N);
            rowsReported = rowsDone;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rowsReported != N) {
        printf("  Progress: %d/%d triangles\n", N, N);
    }

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaFreeHost(hCp[s]));
        CUDA_CHECK(cudaFree(dCp[s]));
        CUDA_CHECK(cudaFree(dScratch[s]));
        CUDA_CHECK(cudaFree(dList[s]));
        CUDA_CHECK(cudaEventDestroy(h2dDone[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFree(dSegCount));
    CUDA_CHECK(cudaFree(dSegStart));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    // Tau(i, j) = ceil(|center_i - center_j| / WAVE_SPEED) is evaluated on the fly
    // by the propagation kernel from the triangle centers already resident on
    // the GPU, which is cheaper than streaming an N x N integer matrix.
    (void)state;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const size_t T = state.numTimesteps;
    GpuState& g = state.gpu;

    const int blocks = (N + SIM_ROWS - 1) / SIM_ROWS;
    for (size_t t = 0; t < T; ++t) {
        kernelSimStep<<<blocks, SIM_THREADS>>>(static_cast<int>(t), N, g.kijT, g.centers,
                                           g.areas, g.rho, g.radE, g.radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    GpuState& g = state.gpu;

    if (T > 0) {
        int threads = std::min(256, ((T + 31) / 32) * 32);
        size_t shmem = 2 * static_cast<size_t>(T) * sizeof(float);
        if (shmem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(kernelDistances,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(shmem)));
        }
        kernelDistances<<<N, threads, shmem>>>(N, T, static_cast<int>(state.sourceIndex),
                                               g.radB, g.distances);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(state.distances.data(), g.distances, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// Fetch the full radiosity history and the form factor matrix for validation
void downloadForValidation(SimulationState& state) {
    const size_t N = state.numTriangles;
    GpuState& g = state.gpu;
    CUDA_CHECK(cudaMemcpy(state.radB.data(), g.radB, state.numTimesteps * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<float> kijT(N * N);
    CUDA_CHECK(cudaMemcpy(kijT.data(), g.kijT, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    state.kij.resize(N * N);
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            state.kij[state.idx2d(i, j)] = kijT[j * N + i];
        }
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

    // Initialize the GPU context early so it is not charged to the timed phases
    CUDA_CHECK(cudaFree(0));

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
    int rc = 0;
    if (validate) {
        downloadForValidation(state);
        if (!validateResults(state)) {
            rc = 1;
        }
    }

    freeGpu(state.gpu);
    return rc;
}
