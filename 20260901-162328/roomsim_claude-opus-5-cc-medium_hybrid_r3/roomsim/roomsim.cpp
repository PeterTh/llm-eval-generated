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
 * ----------------------------------------------------------------------------
 * Parallelization strategy
 * ----------------------------------------------------------------------------
 * The dominant cost (>98%) is the form factor computation: for every ordered
 * pair of triangles, NUM_RAYS random visibility rays are traced through an
 * octree.  That work is offloaded to CUDA devices:
 *
 *  - MPI distributes the triangle rows (the "i" index of every N x N matrix)
 *    across ranks; one GPU per rank.  Row blocks are the unit of distribution
 *    for Kij, Tau, the radiosity update and the cross-correlation.
 *  - CUDA computes Kij (one warp = 2 triangle pairs = 2 x 16 rays), Tau, the
 *    per-timestep radiosity update and the cross-correlation.
 *  - OpenMP drives all host-side work: octree construction, the visibility
 *    culling pass, the random number stream generation that feeds the GPU
 *    pipeline, and the validation reductions.
 *
 * Bit-level fidelity to the sequential reference is preserved where it is
 * observable in the results:
 *  - The Mersenne twister stream is reproduced *exactly*, including the order
 *    in which pairs consume it.  Every host thread jumps directly to its
 *    offset in the stream (GF(2) polynomial jump-ahead), so the stream is
 *    generated in parallel without changing a single value.
 *  - All floating point reductions (per-ray form factor sum, radiosity sum
 *    over j, cross correlation sums) keep the sequential summation order.
 *  - The geometry expressions are written with explicit single rounding
 *    operations (see RS_FMA below) that reproduce the fused multiply-add chains
 *    the reference build ends up with, so host and device agree bit for bit.
 *  - Octree traversal order is irrelevant because visibility is a logical OR
 *    over the candidate triangles, so the device uses a stackless traversal.
 *
 * The results are identical, bit for bit, to the sequential implementation for
 * every mesh size (verified up to 20480 triangles), independent of the number
 * of ranks and threads.
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

#include <sched.h>
#include <unistd.h>

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

// Random numbers consumed by one non-culled triangle pair (NUM_RAYS * 2 points * 2 coords)
constexpr int RND_PER_PAIR = 4 * NUM_RAYS;
constexpr uint32_t RND_CULLED = 0xffffffffu;

// ============================================================================
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ============================================================================
// Single rounding primitives
// ============================================================================
//
// The reference implementation is compiled with -ffp-contract=fast, so its
// geometry expressions end up as particular fused multiply-add chains.  Since
// the results of this benchmark are chaotic with respect to last bit changes
// (the cross correlation picks an argmax), the expressions that feed the
// results are spelled out here with explicit single rounding operations that
// reproduce the reference arithmetic operation for operation, on the host as
// well as on the device.
#ifdef __CUDA_ARCH__
#define RS_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define RS_MUL(a, b) __fmul_rn((a), (b))
#define RS_ADD(a, b) __fadd_rn((a), (b))
#define RS_DIV(a, b) __fdiv_rn((a), (b))
#else
#define RS_FMA(a, b, c) fmaf((a), (b), (c))
#define RS_MUL(a, b) ((a) * (b))
#define RS_ADD(a, b) ((a) + (b))
#define RS_DIV(a, b) ((a) / (b))
#endif

// a*b - c*d, with the first product fused
__host__ __device__ inline val_t rsMulSubA(val_t a, val_t b, val_t c, val_t d) {
    return RS_FMA(a, b, -RS_MUL(c, d));
}

// a*b - c*d, with the second product fused (negated multiply-add)
__host__ __device__ inline val_t rsMulSubB(val_t a, val_t b, val_t c, val_t d) {
    return RS_FMA(-c, d, RS_MUL(a, b));
}

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
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = normRef();
        return n > EPSILON ? Vec3(RS_DIV(x, n), RS_DIV(y, n), RS_DIV(z, n)) : Vec3();
    }

    // Same values as cross()/squaredNorm()/dot(), but with the rounding of the
    // reference build pinned down (see RS_FMA above).  The reference happens to
    // fuse the cross product differently in the two places where it is used,
    // hence the two flavours.
    __host__ __device__ Vec3 crossRefNormal(const Vec3& o) const {
        return {rsMulSubB(y, o.z, z, o.y), rsMulSubB(z, o.x, x, o.z), rsMulSubA(x, o.y, y, o.x)};
    }
    __host__ __device__ Vec3 crossRef(const Vec3& o) const {
        return {rsMulSubA(y, o.z, z, o.y), rsMulSubA(z, o.x, x, o.z), rsMulSubA(x, o.y, y, o.x)};
    }
    __host__ __device__ val_t squaredNormRef() const {
        return RS_FMA(z, z, RS_FMA(x, x, RS_MUL(y, y)));
    }
    __host__ __device__ val_t normRef() const { return std::sqrt(squaredNormRef()); }
    __host__ __device__ val_t dotRef(const Vec3& o) const {
        return RS_FMA(z, o.z, RS_FMA(x, o.x, RS_MUL(y, o.y)));
    }
    __host__ __device__ Vec3 normalizedRef() const {
        val_t n = normRef();
        return n > EPSILON ? Vec3(RS_DIV(x, n), RS_DIV(y, n), RS_DIV(z, n)) : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).crossRefNormal(c - a).normalizedRef()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.crossRef(ac).normRef();
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

// Subtrees with at least this many triangles are built by a separate OpenMP task
constexpr size_t OCTREE_TASK_CUTOFF = 512;

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

        // The recursion is spread over the OpenMP team; the resulting tree is
        // identical to the sequential one (each subtree only depends on its
        // own index list and bounds).
        #pragma omp parallel
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
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
                if (childIndices[i].size() >= OCTREE_TASK_CUTOFF) {
                    Octree* child = children[i].get();
                    const std::vector<size_t>* ci = &childIndices[i];
                    #pragma omp task firstprivate(child, ci, childMin, childMax)
                    child->buildNode(*ci, childMin, childMax);
                } else {
                    children[i]->buildNode(childIndices[i], childMin, childMax);
                }
            }
        }
        // childIndices must outlive the tasks that reference it
        #pragma omp taskwait
    }
};

// ============================================================================
// Flattened octree for GPU traversal
// ============================================================================

// Depth first pre-order layout with "miss" links (stackless traversal).  The
// traversal visits exactly the same set of triangles as the recursive host
// version; the order differs, which cannot change the result because
// visibility is the logical OR over all candidate triangles.
struct FlatOctree {
    std::vector<float4> geo;      // 2 entries per node: (cx,cy,cz,hx), (hy,hz,0,0)
    std::vector<uint4> info;      // (miss, triFirst, triCount, 0)
    std::vector<uint32_t> tris;   // concatenated leaf triangle indices

    uint32_t flatten(const Octree* node) {
        uint32_t self = static_cast<uint32_t>(info.size());
        geo.push_back(make_float4(node->center.x, node->center.y, node->center.z, node->halfExtent.x));
        geo.push_back(make_float4(node->halfExtent.y, node->halfExtent.z, 0.0f, 0.0f));
        info.push_back(make_uint4(0, 0, 0, 0));

        if (!node->triangleIndices.empty()) {
            uint32_t first = static_cast<uint32_t>(tris.size());
            for (size_t idx : node->triangleIndices) tris.push_back(static_cast<uint32_t>(idx));
            info[self].y = first;
            info[self].z = static_cast<uint32_t>(node->triangleIndices.size());
        } else {
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) flatten(node->children[i].get());
            }
        }
        info[self].x = static_cast<uint32_t>(info.size());  // miss link: end of subtree
        return self;
    }
};

struct DevScene {
    const float4* __restrict__ triV;    // 3 vertices per triangle
    const float4* __restrict__ triN;    // triangle normal
    const float4* __restrict__ nodeGeo;
    const uint4* __restrict__ nodeInfo;
    const uint32_t* __restrict__ nodeTris;
    uint32_t numNodes;
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

        // Build triangles (normals pointing inward for a "room").  Kept
        // sequential: the face normals are computed here and even a change of
        // vectorization would perturb them in the last bit.
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

// Bit-exact reimplementation of std::mt19937 as used through
// std::uniform_real_distribution<float>(0,1) by the reference implementation,
// extended with GF(2) jump-ahead so that any thread can start generating at an
// arbitrary position of the (single, global) random stream.
namespace mt {

constexpr int N = 624;
constexpr int M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0dfu;
constexpr uint32_t UPPER_MASK = 0x80000000u;
constexpr uint32_t LOWER_MASK = 0x7fffffffu;

constexpr int PHI_DEG = 19937;              // degree of the characteristic polynomial
constexpr int M_DEG = PHI_DEG + 31;         // degree of the annihilating polynomial x^31*phi
constexpr int RW = (M_DEG + 63) / 64;       // 64-bit words of a residue (degree < M_DEG)
constexpr int MW = RW + 2;                  // words of the (shifted) modulus

// Shifted copies of the modulus, built once per process.
struct JumpTables {
    uint64_t mshift[64][MW];
    bool ready = false;
};
static JumpTables gTables;

static inline int getBit(const uint64_t* a, int i) { return static_cast<int>((a[i >> 6] >> (i & 63)) & 1ull); }
static inline void setBit(uint64_t* a, int i) { a[i >> 6] |= 1ull << (i & 63); }

// Berlekamp-Massey over GF(2): minimal connection polynomial of a bit sequence
static int berlekampMassey(const uint8_t* s, int len, uint64_t* C, int words) {
    std::vector<uint64_t> B(words, 0), T(words, 0), R(words, 0);
    std::memset(C, 0, static_cast<size_t>(words) * 8);
    C[0] = 1;
    B[0] = 1;
    int L = 0, m = 1;
    for (int n = 0; n < len; ++n) {
        // discrepancy = s[n] ^ sum_{i=1..L} C_i * s[n-i];  R holds s[n-1], s[n-2], ...
        uint64_t acc = 0;
        const int nw = (L >> 6) + 1;
        for (int w = 0; w < nw; ++w) {
            uint64_t cs = (C[w] >> 1) | (w + 1 < words ? (C[w + 1] << 63) : 0ull);
            acc ^= cs & R[w];
        }
        const int d = s[n] ^ (__builtin_popcountll(acc) & 1);
        if (d) {
            std::memcpy(T.data(), C, static_cast<size_t>(words) * 8);
            const int wo = m >> 6, bo = m & 63;
            for (int w = words - 1; w >= 0; --w) {
                uint64_t v = 0;
                if (w - wo >= 0) v = B[w - wo] << bo;
                if (bo && w - wo - 1 >= 0) v |= B[w - wo - 1] >> (64 - bo);
                C[w] ^= v;
            }
            if (2 * L <= n) {
                L = n + 1 - L;
                std::memcpy(B.data(), T.data(), static_cast<size_t>(words) * 8);
                m = 1;
            } else {
                ++m;
            }
        } else {
            ++m;
        }
        for (int w = words - 1; w > 0; --w) R[w] = (R[w] << 1) | (R[w - 1] >> 63);
        R[0] = (R[0] << 1) | static_cast<uint64_t>(s[n]);
    }
    return L;
}

// t (tw words) modulo the stored modulus, in place
static void polyReduce(uint64_t* t, int tw) {
    for (int k = tw * 64 - 1; k >= M_DEG; --k) {
        if (!getBit(t, k)) continue;
        const int sh = k - M_DEG;
        const uint64_t* ms = gTables.mshift[sh & 63];
        const int wo = sh >> 6;
        const int lim = MW < tw - wo ? MW : tw - wo;
        uint64_t* dst = t + wo;
        for (int w = 0; w < lim; ++w) dst[w] ^= ms[w];
    }
}

static const uint64_t kSpread[16] = {
    0x0000ull, 0x0001ull, 0x0004ull, 0x0005ull, 0x0010ull, 0x0011ull, 0x0014ull, 0x0015ull,
    0x0040ull, 0x0041ull, 0x0044ull, 0x0045ull, 0x0050ull, 0x0051ull, 0x0054ull, 0x0055ull};

static inline uint64_t spread32(uint32_t v) {
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) r |= kSpread[(v >> (4 * i)) & 15] << (8 * i);
    return r;
}

static void polySquareMod(uint64_t* r) {
    uint64_t t[2 * RW + 4];
    std::memset(t, 0, sizeof(t));
    for (int w = 0; w < RW; ++w) {
        t[2 * w] = spread32(static_cast<uint32_t>(r[w] & 0xffffffffu));
        t[2 * w + 1] = spread32(static_cast<uint32_t>(r[w] >> 32));
    }
    polyReduce(t, 2 * RW);
    std::memcpy(r, t, RW * 8);
}

static void polyMulXMod(uint64_t* r) {
    uint64_t t[RW + 2];
    std::memset(t, 0, sizeof(t));
    t[RW] = r[RW - 1] >> 63;
    for (int w = RW - 1; w > 0; --w) t[w] = (r[w] << 1) | (r[w - 1] >> 63);
    t[0] = r[0] << 1;
    polyReduce(t, RW + 1);
    std::memcpy(r, t, RW * 8);
}

// State in "linear" form: the 624 words preceding the next output, rotated by pos
struct LinState {
    uint32_t s[N];
    int pos;
    void zero() { std::memset(s, 0, sizeof(s)); pos = 0; }
    void advance() {
        const int i1 = pos + 1 >= N ? pos + 1 - N : pos + 1;
        const int im = pos + M >= N ? pos + M - N : pos + M;
        const uint32_t y = (s[pos] & UPPER_MASK) | (s[i1] & LOWER_MASK);
        s[pos] = s[im] ^ (y >> 1) ^ ((s[i1] & 1u) ? MATRIX_A : 0u);
        pos = i1;
    }
    void xorFrom(const LinState& o) {
        for (int k = 0; k < N; ++k) {
            int a = pos + k; if (a >= N) a -= N;
            int b = o.pos + k; if (b >= N) b -= N;
            s[a] ^= o.s[b];
        }
    }
};

// Build the annihilating polynomial x^31 * phi(x) of the word advance map.
// phi is recovered with Berlekamp-Massey from a bit sequence of the generator
// (phi is primitive, so any non-degenerate sequence yields it exactly).
static void buildTables();

struct Generator {
    uint32_t s[N];
    int p;

    void seed(uint32_t sd) {
        s[0] = sd;
        for (uint32_t i = 1; i < N; ++i) s[i] = 1812433253u * (s[i - 1] ^ (s[i - 1] >> 30)) + i;
        p = N;
    }

    void twist() {
        for (int i = 0; i < N - M; ++i) {
            const uint32_t y = (s[i] & UPPER_MASK) | (s[i + 1] & LOWER_MASK);
            s[i] = s[i + M] ^ (y >> 1) ^ ((s[i + 1] & 1u) ? MATRIX_A : 0u);
        }
        for (int i = N - M; i < N - 1; ++i) {
            const uint32_t y = (s[i] & UPPER_MASK) | (s[i + 1] & LOWER_MASK);
            s[i] = s[i - (N - M)] ^ (y >> 1) ^ ((s[i + 1] & 1u) ? MATRIX_A : 0u);
        }
        const uint32_t y = (s[N - 1] & UPPER_MASK) | (s[0] & LOWER_MASK);
        s[N - 1] = s[M - 1] ^ (y >> 1) ^ ((s[0] & 1u) ? MATRIX_A : 0u);
        p = 0;
    }

    static uint32_t temper(uint32_t z) {
        z ^= (z >> 11);
        z ^= (z << 7) & 0x9d2c5680u;
        z ^= (z << 15) & 0xefc60000u;
        z ^= (z >> 18);
        return z;
    }

    inline uint32_t nextRaw() {
        if (p >= N) twist();
        return s[p++];
    }

    // Identical to std::uniform_real_distribution<float>(0,1)(std::mt19937)
    inline val_t next() {
        const val_t u = static_cast<val_t>(temper(nextRaw())) * (1.0f / 4294967296.0f);
        return u >= 1.0f ? 0.99999994f : u;  // std::nextafter(1,0)
    }

    // Advance the stream by n outputs without generating them
    void skip(uint64_t n) {
        while (n > 0) {
            if (p >= N) twist();
            const uint64_t take = std::min<uint64_t>(static_cast<uint64_t>(N - p), n);
            p += static_cast<int>(take);
            n -= take;
        }
    }

    // Advance the stream by n outputs (state must be at a twist boundary).
    // Short distances are cheaper to walk than to jump.
    void jump(uint64_t n) {
        if (n == 0) return;
        if (n < (1ull << 25)) {
            skip(n);
            return;
        }
        buildTables();
        if (p != N) {  // normalize: fold pending outputs into the jump distance
            n += static_cast<uint64_t>(N - p);
            p = N;
        }
        uint64_t q[RW];
        std::memset(q, 0, sizeof(q));
        q[0] = 1;
        int top = 63;
        while (top > 0 && !((n >> top) & 1ull)) --top;
        for (int b = top; b >= 0; --b) {
            polySquareMod(q);
            if ((n >> b) & 1ull) polyMulXMod(q);
        }
        LinState h, r;
        std::memcpy(h.s, s, sizeof(h.s));
        h.pos = 0;
        r.zero();
        for (int i = M_DEG - 1; i >= 0; --i) {
            r.advance();
            if (getBit(q, i)) r.xorFrom(h);
        }
        for (int k = 0; k < N; ++k) {
            int a = r.pos + k; if (a >= N) a -= N;
            s[k] = r.s[a];
        }
        p = N;
    }
};

static void buildTables() {
    if (gTables.ready) return;
    const int len = 2 * PHI_DEG + 64;
    std::vector<uint8_t> seq(len);
    Generator g;
    g.seed(5489u);
    for (int i = 0; i < 31; ++i) g.nextRaw();  // skip the transient subspace
    for (int i = 0; i < len; ++i) seq[i] = static_cast<uint8_t>(g.nextRaw() & 1u);

    std::vector<uint64_t> C(RW + 4, 0);
    const int L = berlekampMassey(seq.data(), len, C.data(), RW + 4);
    if (L != PHI_DEG) {
        fprintf(stderr, "internal error: MT characteristic polynomial degree %d\n", L);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::memset(gTables.mshift, 0, sizeof(gTables.mshift));
    uint64_t* m0 = gTables.mshift[0];
    for (int j = 0; j <= L; ++j) {
        if (getBit(C.data(), L - j)) setBit(m0, j + 31);  // x^31 * reverse(C)
    }
    for (int b = 1; b < 64; ++b) {
        for (int w = MW - 1; w > 0; --w) gTables.mshift[b][w] = (m0[w] << b) | (m0[w - 1] >> (64 - b));
        gTables.mshift[b][0] = m0[0] << b;
    }
    gTables.ready = true;
}

}  // namespace mt

// ============================================================================
// Device side geometry / ray casting (expression order kept identical to the
// sequential reference so that the floating point results match)
// ============================================================================

__device__ inline Vec3 loadVec(const float4* p, size_t i) {
    const float4 v = p[i];
    return Vec3(v.x, v.y, v.z);
}

// Generate a random point inside a triangle using barycentric coordinates.
// a + ab*u + ac*v, fused exactly like the reference build.
__device__ inline Vec3 randomPointInTriangleDev(const Vec3& a, const Vec3& ab, const Vec3& ac,
                                                val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return Vec3(RS_FMA(v, ac.x, RS_FMA(u, ab.x, a.x)),
                RS_FMA(v, ac.y, RS_FMA(u, ab.y, a.y)),
                RS_FMA(v, ac.z, RS_FMA(u, ab.z, a.z)));
}

// Ray-Triangle Intersection (Moeller-Trumbore algorithm)
__device__ inline val_t rayTriangleIntersectDev(const Vec3& orig, const Vec3& dir,
                                                const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.crossRef(e2);
    val_t det = e1.dotRef(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = RS_DIV(1.0f, det);
    Vec3 tvec = orig - v0;
    val_t u = RS_MUL(tvec.dotRef(pvec), invDet);
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.crossRef(e1);
    val_t v = RS_MUL(dir.dotRef(qvec), invDet);
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return RS_MUL(e2.dotRef(qvec), invDet);
}

__device__ inline bool rayIntersectsBoxDev(const Vec3& p1, const Vec3& p2,
                                           const Vec3& center, const Vec3& halfExtent) {
    const Vec3 d = Vec3(RS_MUL(p2.x - p1.x, 0.5f), RS_MUL(p2.y - p1.y, 0.5f), RS_MUL(p2.z - p1.z, 0.5f));
    const Vec3 c = Vec3(RS_ADD(p1.x, d.x) - center.x, RS_ADD(p1.y, d.y) - center.y,
                        RS_ADD(p1.z, d.z) - center.z);
    const Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

    if (std::abs(c.x) > RS_ADD(halfExtent.x, ad.x)) return false;
    if (std::abs(c.y) > RS_ADD(halfExtent.y, ad.y)) return false;
    if (std::abs(c.z) > RS_ADD(halfExtent.z, ad.z)) return false;

    if (std::abs(rsMulSubA(d.y, c.z, d.z, c.y)) >
        RS_ADD(RS_FMA(halfExtent.z, ad.y, RS_MUL(halfExtent.y, ad.z)), EPSILON)) return false;
    if (std::abs(rsMulSubA(d.z, c.x, d.x, c.z)) >
        RS_ADD(RS_FMA(halfExtent.x, ad.z, RS_MUL(halfExtent.z, ad.x)), EPSILON)) return false;
    if (std::abs(rsMulSubA(d.x, c.y, d.y, c.x)) >
        RS_ADD(RS_FMA(halfExtent.y, ad.x, RS_MUL(halfExtent.x, ad.y)), EPSILON)) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Stackless octree traversal; the root node is always entered (as in the
// recursive reference), every other node is box tested.
__device__ inline bool isRayBlockedDev(const Vec3& from, const Vec3& to, const DevScene& sc,
                                       uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.normRef();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = Vec3(RS_DIV(dir.x, rayLen), RS_DIV(dir.y, rayLen), RS_DIV(dir.z, rayLen));

    uint32_t node = 0;
    while (node < sc.numNodes) {
        const uint4 inf = sc.nodeInfo[node];
        if (node != 0) {
            const float4 g0 = sc.nodeGeo[2 * node];
            const float4 g1 = sc.nodeGeo[2 * node + 1];
            if (!rayIntersectsBoxDev(from, to, Vec3(g0.x, g0.y, g0.z), Vec3(g0.w, g1.x, g1.y))) {
                node = inf.x;
                continue;
            }
        }
        if (inf.z != 0u) {  // leaf
            const uint32_t first = inf.y;
            for (uint32_t k = 0; k < inf.z; ++k) {
                const uint32_t idx = sc.nodeTris[first + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Vec3 a = loadVec(sc.triV, 3 * static_cast<size_t>(idx) + 0);
                const Vec3 b = loadVec(sc.triV, 3 * static_cast<size_t>(idx) + 1);
                const Vec3 c = loadVec(sc.triV, 3 * static_cast<size_t>(idx) + 2);
                const val_t dist = rayTriangleIntersectDev(from, dirNorm, a, b, c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
            node = inf.x;
        } else {
            node = node + 1;
        }
    }
    return false;
}

// cosPhi(-v, normal).  Negating the (already rounded) dot product is exactly
// what the host code computes for -v, while letting the device contract the
// dot product itself the same way the host compiler does.
__device__ inline val_t cosPhiNegDev(const Vec3& v, val_t vNorm, const Vec3& normal) {
    if (vNorm <= EPSILON) return ZERO;
    const val_t c = RS_DIV(0.0f - v.dotRef(normal), vNorm);
    return c > 0.0f ? c : 0.0f;   // std::max(ZERO, c)
}

__device__ inline val_t cosPhiDev(const Vec3& v, val_t vNorm, const Vec3& normal) {
    if (vNorm <= EPSILON) return ZERO;
    const val_t c = RS_DIV(v.dotRef(normal), vNorm);
    return c > 0.0f ? c : 0.0f;   // std::max(ZERO, c)
}

// ============================================================================
// Form factor kernel: one warp handles two triangle pairs (16 rays each)
// ============================================================================

__global__ void kijKernel(DevScene sc, int numRows, int n, int rowBase,
                          const uint32_t* __restrict__ rowIdx,
                          const uint32_t* __restrict__ rndBase,
                          const float4* __restrict__ rnd,
                          float* __restrict__ kijRows) {
    const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pair = gid >> 4;
    const int ray = static_cast<int>(gid & 15);
    const long long numPairs = static_cast<long long>(numRows) * n;
    const bool valid = pair < numPairs;

    val_t contrib = ZERO;
    uint32_t base = RND_CULLED;
    int i = 0, j = 0;
    if (valid) {
        const int slot = static_cast<int>(pair / n);
        j = static_cast<int>(pair - static_cast<long long>(slot) * n);
        i = static_cast<int>(rowIdx[slot]);
        base = rndBase[pair];
    }

    if (base != RND_CULLED) {
        const Vec3 ia = loadVec(sc.triV, 3 * static_cast<size_t>(i) + 0);
        const Vec3 iab = loadVec(sc.triV, 3 * static_cast<size_t>(i) + 1) - ia;
        const Vec3 iac = loadVec(sc.triV, 3 * static_cast<size_t>(i) + 2) - ia;
        const Vec3 ja = loadVec(sc.triV, 3 * static_cast<size_t>(j) + 0);
        const Vec3 jab = loadVec(sc.triV, 3 * static_cast<size_t>(j) + 1) - ja;
        const Vec3 jac = loadVec(sc.triV, 3 * static_cast<size_t>(j) + 2) - ja;
        const Vec3 nI = loadVec(sc.triN, i);
        const Vec3 nJ = loadVec(sc.triN, j);

        const float4 r = rnd[base + ray];
        const Vec3 pI = randomPointInTriangleDev(ia, iab, iac, r.x, r.y);
        const Vec3 pJ = randomPointInTriangleDev(ja, jab, jac, r.z, r.w);

        if (!isRayBlockedDev(pI, pJ, sc, static_cast<uint32_t>(i), static_cast<uint32_t>(j))) {
            const Vec3 v = pJ - pI;
            const val_t distSqr = v.squaredNormRef();
            if (distSqr >= EPSILON) {
                const val_t vNorm = std::sqrt(distSqr);
                const val_t cosPhiI = cosPhiDev(v, vNorm, nI);
                const val_t cosPhiJ = cosPhiNegDev(v, vNorm, nJ);
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    contrib = RS_DIV(RS_MUL(cosPhiI, cosPhiJ), RS_MUL(PI, distSqr));
                }
            }
        }
    }

    // Sum the 16 rays in the sequential order of the reference implementation
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lbase = lane & 16;
    val_t kij = ZERO;
    #pragma unroll
    for (int k = 0; k < 16; ++k) kij += __shfl_sync(0xffffffffu, contrib, lbase + k, 32);

    // Rows of a round are not contiguous: place the value at its row in the block
    if (valid && ray == 0) kijRows[static_cast<size_t>(i - rowBase) * n + j] = kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau kernel (row major storage: tau[row * N + j])
// ============================================================================

__global__ void tauKernel(const float4* __restrict__ triV, int numRows, int n, int rowBase,
                          int* __restrict__ tau) {
    const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= static_cast<long long>(numRows) * n) return;
    const int r = static_cast<int>(gid / n);
    const int j = static_cast<int>(gid - static_cast<long long>(r) * n);
    const int i = rowBase + r;
    int tauv = 0;
    if (i != j) {
        const Vec3 ci = (loadVec(triV, 3 * static_cast<size_t>(i) + 0) +
                         loadVec(triV, 3 * static_cast<size_t>(i) + 1) +
                         loadVec(triV, 3 * static_cast<size_t>(i) + 2)) / 3.0f;
        const Vec3 cj = (loadVec(triV, 3 * static_cast<size_t>(j) + 0) +
                         loadVec(triV, 3 * static_cast<size_t>(j) + 1) +
                         loadVec(triV, 3 * static_cast<size_t>(j) + 2)) / 3.0f;
        const val_t dist = (ci - cj).normRef();
        tauv = static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
    }
    tau[gid] = tauv;
}

// ============================================================================
// Radiosity update: one warp per owned triangle, j-loop in sequential order
// ============================================================================

__global__ void simKernel(int t, int n, int rowBase, int numRows,
                          const float* __restrict__ kij, const int* __restrict__ tau,
                          const float* __restrict__ areas, const float* __restrict__ rho,
                          const float* __restrict__ radE, const float* __restrict__ radB,
                          float* __restrict__ outRow) {
    const int warpId = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warpId >= numRows) return;   // whole warps, so warp wide shuffles stay valid
    const int r = warpId;
    const int i = rowBase + r;

    const float* __restrict__ kRow = kij + static_cast<size_t>(r) * n;
    const int* __restrict__ tRow = tau + static_cast<size_t>(r) * n;

    // The 32 lanes of the warp evaluate 32 consecutive j, but the sum is
    // accumulated strictly in increasing j order (as in the reference).
    val_t sumB = ZERO;
    for (int j0 = 0; j0 < n; j0 += 32) {
        const int j = j0 + lane;
        val_t c = ZERO;
        if (j < n && i != j) {
            const int tauij = tRow[j];
            const val_t kijv = kRow[j];
            if (t >= tauij && kijv > ZERO) {
                const val_t radJ = radB[static_cast<size_t>(t - tauij) * n + j];
                if (radJ > ZERO) {
                    const val_t w = RS_MUL(kijv, areas[j]);          // std::min(w, ONE)
                    c = RS_MUL((ONE < w ? ONE : w), radJ);
                }
            }
        }
        // the reference adds the (separately rounded) products one by one
        #pragma unroll
        for (int m = 0; m < 32; ++m) sumB = RS_ADD(sumB, __shfl_sync(0xffffffffu, c, m, 32));
    }

    // the reference does contract this one
    if (lane == 0) outRow[r] = RS_FMA(rho[i], sumB, radE[static_cast<size_t>(t) * n + i]);
}

// ============================================================================
// Cross correlation kernel: one thread per owned triangle
// ============================================================================

__global__ void crossCorrKernel(int n, int numTimesteps, int rowBase, int numRows,
                                int sourceIndex, const float* __restrict__ radB,
                                float* __restrict__ corr) {
    const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= static_cast<long long>(numRows) * numTimesteps) return;
    const int t = static_cast<int>(gid / numRows);
    const int r = static_cast<int>(gid - static_cast<long long>(t) * numRows);
    const int i = rowBase + r;

    val_t sum = ZERO;
    for (int tt = t; tt < numTimesteps; ++tt) {
        const val_t pB = radB[static_cast<size_t>(tt) * n + i];
        const val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
        sum = RS_ADD(sum, RS_MUL(pS, pB));
    }
    corr[gid] = sum;
}

__global__ void argMaxKernel(int numTimesteps, int numRows, const float* __restrict__ corr,
                             float* __restrict__ distOut) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRows) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numTimesteps; ++t) {
        const val_t sum = corr[static_cast<size_t>(t) * numRows + r];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distOut[r] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// Count non-zero form factors of the owned rows (order independent)
__global__ void countNonZeroKernel(const float* __restrict__ kij, size_t count,
                                   unsigned long long* __restrict__ out) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (; i < count; i += stride) {
        if (kij[i] > EPSILON) ++local;
    }
    typedef unsigned long long ull;
    __shared__ ull red[256];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, red[0]);
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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flat;                // GPU friendly copy of the octree

    size_t sourceIndex;

    // MPI decomposition: this rank owns rows [rowBase, rowBase + numRows)
    int mpiRank = 0, mpiSize = 1;
    int rowBase = 0, numRows = 0;
    std::vector<int> rowCounts, rowOffsets;
    std::vector<uint64_t> cullMask;  // one bit per owned pair: pair is not culled

    // Device resources
    float4* d_triV = nullptr;
    float4* d_triN = nullptr;
    float4* d_nodeGeo = nullptr;
    uint4* d_nodeInfo = nullptr;
    uint32_t* d_nodeTris = nullptr;
    float* d_kij = nullptr;         // row major, [numRows x N]
    int* d_tau = nullptr;           // row major, [numRows x N]
    float* d_corr = nullptr;        // cross correlation, [T x numRows]
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_rowOut = nullptr;
    float* d_dist = nullptr;
    unsigned long long* d_count = nullptr;
    DevScene scene{};

    // Kij pipeline (double buffered host staging + device buffers)
    int numChunks = 0;              // host threads feeding the GPU
    int roundRows = 0;              // rows per pipeline round (<= numChunks)
    std::vector<int> chunkBegin, chunkEnd;
    float* h_rnd[2] = {nullptr, nullptr};
    uint32_t* h_base[2] = {nullptr, nullptr};
    uint32_t* h_row[2] = {nullptr, nullptr};
    float* d_rnd[2] = {nullptr, nullptr};
    uint32_t* d_base[2] = {nullptr, nullptr};
    uint32_t* d_row[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

static void initializeDevice(SimulationState& state) {
    const size_t n = state.numTriangles;

    // Vertices / normals
    std::vector<float4> hv(3 * n), hn(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        hv[3 * i + 0] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        hv[3 * i + 1] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        hv[3 * i + 2] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        hn[i] = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
    }

    CUDA_CHECK(cudaMalloc(&state.d_triV, hv.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_triN, hn.size() * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(state.d_triV, hv.data(), hv.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triN, hn.data(), hn.size() * sizeof(float4), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_nodeGeo, state.flat.geo.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeInfo, state.flat.info.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeTris, std::max<size_t>(1, state.flat.tris.size()) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(state.d_nodeGeo, state.flat.geo.data(),
                          state.flat.geo.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeInfo, state.flat.info.data(),
                          state.flat.info.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    if (!state.flat.tris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_nodeTris, state.flat.tris.data(),
                              state.flat.tris.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    state.scene.triV = state.d_triV;
    state.scene.triN = state.d_triN;
    state.scene.nodeGeo = state.d_nodeGeo;
    state.scene.nodeInfo = state.d_nodeInfo;
    state.scene.nodeTris = state.d_nodeTris;
    state.scene.numNodes = static_cast<uint32_t>(state.flat.info.size());

    const size_t rows = static_cast<size_t>(state.numRows);
    CUDA_CHECK(cudaMalloc(&state.d_kij, std::max<size_t>(1, rows * n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, std::max<size_t>(1, rows * n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, state.numTimesteps * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, state.numTimesteps * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rowOut, std::max<size_t>(1, rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_dist, std::max<size_t>(1, rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_corr, std::max<size_t>(1, rows * state.numTimesteps) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_count, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(),
                          state.numTimesteps * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, state.numTimesteps * n * sizeof(float)));

    // Pipeline buffers: one row per chunk per round
    const size_t rndPerRound = static_cast<size_t>(state.roundRows) * n * RND_PER_PAIR;
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaHostAlloc(&state.h_rnd[b], rndPerRound * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&state.h_base[b], static_cast<size_t>(state.roundRows) * n * sizeof(uint32_t),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&state.h_row[b], static_cast<size_t>(state.roundRows) * sizeof(uint32_t),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&state.d_rnd[b], rndPerRound * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_base[b], static_cast<size_t>(state.roundRows) * n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&state.d_row[b], static_cast<size_t>(state.roundRows) * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&state.stream[b]));
    }
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (state.mpiRank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Row decomposition over MPI ranks
    const int n = static_cast<int>(state.numTriangles);
    state.rowCounts.assign(state.mpiSize, 0);
    state.rowOffsets.assign(state.mpiSize, 0);
    for (int r = 0; r < state.mpiSize; ++r) {
        const int beg = static_cast<int>((static_cast<long long>(n) * r) / state.mpiSize);
        const int end = static_cast<int>((static_cast<long long>(n) * (r + 1)) / state.mpiSize);
        state.rowOffsets[r] = beg;
        state.rowCounts[r] = end - beg;
    }
    state.rowBase = state.rowOffsets[state.mpiRank];
    state.numRows = state.rowCounts[state.mpiRank];

    // Build octree for spatial acceleration
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.flat.flatten(&state.octree);

    // Initialize areas (sequential: keeps the reference's rounding)
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(state.numTriangles, reflectivity);

    // Initialize matrices (Kij / Tau live on the device, distributed by rows)
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Jump-ahead tables of the Mersenne twister (depend only on the generator)
    mt::buildTables();

    // Pipeline geometry: one row per chunk per round.  At most one chunk per
    // host thread, capped so that the staging buffers stay within a sane size
    // and so that the per chunk jump ahead of the random stream (one GF(2)
    // polynomial exponentiation each) stays cheap - a handful of rows per round
    // already saturates the device.
    const int threads = omp_get_max_threads();
    const size_t rndBudget = size_t(256) << 20;  // bytes per staging buffer
    int maxRows = static_cast<int>(rndBudget / (static_cast<size_t>(n) * RND_PER_PAIR * sizeof(float)));
    if (maxRows < 1) maxRows = 1;
    state.numChunks = std::max(1, std::min({threads, state.numRows, maxRows, 64}));
    state.roundRows = state.numChunks;

    state.chunkBegin.assign(state.numChunks, 0);
    state.chunkEnd.assign(state.numChunks, 0);
    for (int c = 0; c < state.numChunks; ++c) {
        const long long rows = state.numRows;
        state.chunkBegin[c] = state.rowBase + static_cast<int>((rows * c) / state.numChunks);
        state.chunkEnd[c] = state.rowBase + static_cast<int>((rows * (c + 1)) / state.numChunks);
    }

    initializeDevice(state);
}

static void releaseSimulation(SimulationState& state) {
    cudaFree(state.d_triV); cudaFree(state.d_triN);
    cudaFree(state.d_nodeGeo); cudaFree(state.d_nodeInfo); cudaFree(state.d_nodeTris);
    cudaFree(state.d_kij); cudaFree(state.d_tau); cudaFree(state.d_corr);
    cudaFree(state.d_areas); cudaFree(state.d_rho);
    cudaFree(state.d_radE); cudaFree(state.d_radB);
    cudaFree(state.d_rowOut); cudaFree(state.d_dist); cudaFree(state.d_count);
    for (int b = 0; b < 2; ++b) {
        cudaFreeHost(state.h_rnd[b]); cudaFreeHost(state.h_base[b]); cudaFreeHost(state.h_row[b]);
        cudaFree(state.d_rnd[b]); cudaFree(state.d_base[b]); cudaFree(state.d_row[b]);
        if (state.stream[b]) cudaStreamDestroy(state.stream[b]);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Cull triangles facing the same direction (identical test, including the
// fused multiply-adds, to the reference implementation).
__host__ __device__ inline bool kijCulled(const Vec3& ni, const Vec3& nj) {
    return ni.dotRef(nj) > 0.99f;
}

void computeFormFactors(SimulationState& state) {
    const int n = static_cast<int>(state.numTriangles);
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    // Visibility culling of the owned rows.  The mask is kept because the
    // random number stream is only consumed by pairs that survive culling, so
    // the mask determines each row's offset in the stream.
    const size_t maskWords = (static_cast<size_t>(n) + 63) / 64;
    std::vector<uint64_t>& mask = state.cullMask;
    mask.assign(maskWords * std::max(1, state.numRows), 0);
    std::vector<uint32_t> perRow(n, 0);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < state.numRows; ++r) {
        const int i = state.rowBase + r;
        const Vec3 ni = state.triangles[i].normal();
        uint64_t* row = mask.data() + static_cast<size_t>(r) * maskWords;
        uint32_t cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (i == j || kijCulled(ni, state.triangles[j].normal())) continue;
            row[j >> 6] |= 1ull << (j & 63);
            ++cnt;
        }
        perRow[i] = cnt;
    }
    if (state.mpiSize > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, state.numRows, MPI_UINT32_T, perRow.data(),
                       state.rowCounts.data(), state.rowOffsets.data(), MPI_UINT32_T, MPI_COMM_WORLD);
    }

    // Position of every row in the global random stream: the reference consumes
    // RND_PER_PAIR values for each non-culled pair, in row major order.
    std::vector<uint64_t> rowRnd(static_cast<size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        rowRnd[static_cast<size_t>(i) + 1] =
            rowRnd[static_cast<size_t>(i)] + static_cast<uint64_t>(perRow[i]) * RND_PER_PAIR;
    }

    // Per-chunk generators, each jumping directly to its position in the stream
    const int C = state.numChunks;
    std::vector<mt::Generator> gens(C);
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < C; ++c) {
        gens[c].seed(42);
        gens[c].jump(rowRnd[static_cast<size_t>(state.chunkBegin[c])]);
    }

    int maxRounds = 0;
    for (int c = 0; c < C; ++c) maxRounds = std::max(maxRounds, state.chunkEnd[c] - state.chunkBegin[c]);

    const int threadsPerBlock = 256;
    std::vector<int> slotChunk(C);
    for (int q = 0; q < maxRounds; ++q) {
        const int b = q & 1;
        if (q >= 2) CUDA_CHECK(cudaStreamSynchronize(state.stream[b]));

        // Compact the chunks that still have a row in this round
        int R = 0;
        for (int c = 0; c < C; ++c) {
            if (state.chunkBegin[c] + q < state.chunkEnd[c]) {
                slotChunk[R] = c;
                state.h_row[b][R] = static_cast<uint32_t>(state.chunkBegin[c] + q);
                ++R;
            }
        }

        // Generate this round's slice of the random stream in parallel
        #pragma omp parallel for schedule(static, 1)
        for (int s = 0; s < R; ++s) {
            const int c = slotChunk[s];
            const int i = state.chunkBegin[c] + q;
            mt::Generator& g = gens[c];
            uint32_t* baseRow = state.h_base[b] + static_cast<size_t>(s) * n;
            float* rnd = state.h_rnd[b] + static_cast<size_t>(s) * n * RND_PER_PAIR;
            const uint64_t* cull = state.cullMask.data() +
                                   static_cast<size_t>(i - state.rowBase) * maskWords;
            uint32_t packed = 0;
            for (int j = 0; j < n; ++j) {
                if (!((cull[j >> 6] >> (j & 63)) & 1ull)) {
                    baseRow[j] = RND_CULLED;
                    continue;
                }
                float* dst = rnd + static_cast<size_t>(packed) * RND_PER_PAIR;
                for (int k = 0; k < RND_PER_PAIR; ++k) dst[k] = g.next();
                // offset in float4 units, relative to the start of the buffer
                baseRow[j] = static_cast<uint32_t>((static_cast<size_t>(s) * n + packed) * (RND_PER_PAIR / 4));
                ++packed;
            }
        }

        const size_t pairs = static_cast<size_t>(R) * n;
        CUDA_CHECK(cudaMemcpyAsync(state.d_row[b], state.h_row[b], static_cast<size_t>(R) * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, state.stream[b]));
        CUDA_CHECK(cudaMemcpyAsync(state.d_base[b], state.h_base[b], pairs * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, state.stream[b]));
        CUDA_CHECK(cudaMemcpyAsync(state.d_rnd[b], state.h_rnd[b],
                                   pairs * RND_PER_PAIR * sizeof(float),
                                   cudaMemcpyHostToDevice, state.stream[b]));

        const long long threads = static_cast<long long>(pairs) * NUM_RAYS;
        const int blocks = static_cast<int>((threads + threadsPerBlock - 1) / threadsPerBlock);
        kijKernel<<<blocks, threadsPerBlock, 0, state.stream[b]>>>(
            state.scene, R, n, state.rowBase, state.d_row[b], state.d_base[b],
            reinterpret_cast<const float4*>(state.d_rnd[b]), state.d_kij);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (state.mpiRank == 0) {
        for (int i = 0; i < n; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %d/%d triangles\n", i + 1, n);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    if (state.numRows == 0) return;
    const size_t total = static_cast<size_t>(state.numRows) * n;
    const int tpb = 256;
    const int blocks = static_cast<int>((total + tpb - 1) / tpb);
    tauKernel<<<blocks, tpb>>>(state.d_triV, state.numRows, n, state.rowBase, state.d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);

    std::vector<float> hostRow(std::max(1, state.numRows));
    const int tpb = 128;
    const int blocks = std::max(1, (state.numRows * 32 + tpb - 1) / tpb);
    // tau >= 1 for every off diagonal pair, so timestep t only reads rows < t:
    // with a single rank the update can write straight into radB and the whole
    // time loop runs on the device without host synchronisation.
    const bool singleRank = state.mpiSize == 1;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        float* out = singleRank ? state.d_radB + static_cast<size_t>(t) * n : state.d_rowOut;
        if (state.numRows > 0) {
            simKernel<<<blocks, tpb>>>(static_cast<int>(t), n, state.rowBase, state.numRows,
                                       state.d_kij, state.d_tau, state.d_areas, state.d_rho,
                                       state.d_radE, state.d_radB, out);
            CUDA_CHECK(cudaGetLastError());
        }
        if (!singleRank) {
            CUDA_CHECK(cudaMemcpy(hostRow.data(), state.d_rowOut, state.numRows * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            val_t* row = &state.radB[state.idxTN(t, 0)];
            MPI_Allgatherv(hostRow.data(), state.numRows, MPI_FLOAT, row,
                           state.rowCounts.data(), state.rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(state.d_radB + static_cast<size_t>(t) * n, row, n * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    if (singleRank) {
        // radB is needed on the host for the validation output
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                              state.numTimesteps * n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);

    std::vector<float> local(std::max(1, state.numRows));
    if (state.numRows > 0) {
        const int tpb = 128;
        const size_t work = static_cast<size_t>(state.numRows) * state.numTimesteps;
        crossCorrKernel<<<static_cast<int>((work + tpb - 1) / tpb), tpb>>>(
            n, static_cast<int>(state.numTimesteps), state.rowBase, state.numRows,
            static_cast<int>(state.sourceIndex), state.d_radB, state.d_corr);
        CUDA_CHECK(cudaGetLastError());
        argMaxKernel<<<(state.numRows + tpb - 1) / tpb, tpb>>>(
            static_cast<int>(state.numTimesteps), state.numRows, state.d_corr, state.d_dist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), state.d_dist, state.numRows * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    if (state.mpiSize > 1) {
        MPI_Allgatherv(local.data(), state.numRows, MPI_FLOAT, state.distances.data(),
                       state.rowCounts.data(), state.rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
    } else {
        std::memcpy(state.distances.data(), local.data(), static_cast<size_t>(state.numRows) * sizeof(float));
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    // Non-zero form factors of the owned rows (order independent reduction)
    unsigned long long localNonZeroKij = 0;
    if (state.numRows > 0) {
        const size_t count = static_cast<size_t>(state.numRows) * state.numTriangles;
        CUDA_CHECK(cudaMemset(state.d_count, 0, sizeof(unsigned long long)));
        const int tpb = 256;
        const int blocks = static_cast<int>(std::min<size_t>(1024, (count + tpb - 1) / tpb));
        countNonZeroKernel<<<blocks, tpb>>>(state.d_kij, count, state.d_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&localNonZeroKij, state.d_count, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    unsigned long long totalNonZeroKij = localNonZeroKij;
    if (state.mpiSize > 1) {
        MPI_Allreduce(&localNonZeroKij, &totalNonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    if (state.mpiRank != 0) return true;

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
    const size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           totalNonZeroKij, totalKij,
           100.0f * static_cast<val_t>(totalNonZeroKij) / static_cast<val_t>(totalKij));

    if (totalNonZeroKij == 0) {
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

// Pick this rank's GPU and size the OpenMP team so that ranks sharing a node
// do not oversubscribe the cores.
static void setupHybridEnvironment(int rank, int size) {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localSize);
    MPI_Comm_free(&local);

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context now

    if (!getenv("OMP_NUM_THREADS")) {
        // Fair share of the node's cores for this rank.  Launchers commonly
        // bind a rank to a single core (which would leave the OpenMP team with
        // nowhere to run), so widen the affinity mask when it is narrower than
        // this rank's share.  Setting OMP_NUM_THREADS keeps full manual control.
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        const int share = std::max(1, static_cast<int>(online / std::max(1, localSize)));
        if (omp_get_num_procs() < share) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (long cpu = 0; cpu < online && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
        omp_set_num_threads(share);
    }
    (void)size;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    setupHybridEnvironment(mpiRank, mpiSize);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mpiSize, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    MPI_Allreduce(MPI_IN_PLACE, &preDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    MPI_Allreduce(MPI_IN_PLACE, &simDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    MPI_Allreduce(MPI_IN_PLACE, &distDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    if (mpiRank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
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
    }

    // Validation
    int failed = 0;
    if (validate) {
        if (!validateResults(state)) failed = 1;
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    releaseSimulation(state);
    MPI_Finalize();
    return failed;
}
