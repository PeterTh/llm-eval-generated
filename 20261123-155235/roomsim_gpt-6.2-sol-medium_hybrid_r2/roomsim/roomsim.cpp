/**
 * Room Response Simulation Benchmark
 *
 * This is a hybrid MPI, OpenMP, and CUDA implementation of room impulse response
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
#include <cinttypes>
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
constexpr val_t ROOM_RADIUS = 10.0f;
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

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    // Fix the device contraction order used by the benchmark's geometry arithmetic.
    __host__ __device__ val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return fmaf(z, o.z, fmaf(x, o.x, __fmul_rn(y, o.y)));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {fmaf(y, o.z, -__fmul_rn(z, o.y)), fmaf(z, o.x, -__fmul_rn(x, o.z)),
                fmaf(x, o.y, -__fmul_rn(y, o.x))};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    __host__ __device__ val_t squaredNorm() const {
#ifdef __CUDA_ARCH__
        return dot(*this);
#else
        return x * x + y * y + z * z;
#endif
    }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
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
// CUDA Storage and Geometry
// ============================================================================

void checkCuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template<typename T, bool Pinned = false>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count = 0) {
        if (count == 0) return;
        if constexpr (Pinned) CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
        else CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    ~CudaBuffer() {
        if (!ptr) return;
        if constexpr (Pinned) CUDA_CHECK(cudaFreeHost(ptr));
        else CUDA_CHECK(cudaFree(ptr));
    }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    T* data() const { return ptr; }
private:
    T* ptr = nullptr;
};

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~CudaStream() { CUDA_CHECK(cudaStreamDestroy(stream)); }
    operator cudaStream_t() const { return stream; }
private:
    cudaStream_t stream;
};

struct TriangleGeometry {
    Vec3 a, ab, ac;
    Vec3 normal, center;
    val_t area;
};

struct OctreeNode {
    Vec3 center, halfExtent;
    idx_t escape;
    idx_t first, count;
};

void flattenOctree(const Octree& tree, std::vector<OctreeNode>& nodes,
                   std::vector<idx_t>& indices) {
    const size_t node = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<idx_t>(indices.size()),
                     static_cast<idx_t>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[node].escape = static_cast<idx_t>(nodes.size());
}

__device__ bool intersectsBox(const OctreeNode& node, const Vec3& midpoint,
                              const Vec3& d, const Vec3& ad) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    return fabsf(c.x) <= h.x + ad.x && fabsf(c.y) <= h.y + ad.y && fabsf(c.z) <= h.z + ad.z &&
           fabsf(d.y * c.z - d.z * c.y) <= h.y * ad.z + h.z * ad.y + EPSILON &&
           fabsf(d.z * c.x - d.x * c.z) <= h.z * ad.x + h.x * ad.z + EPSILON &&
           fabsf(d.x * c.y - d.y * c.x) <= h.x * ad.y + h.y * ad.x + EPSILON;
}

__device__ bool intersectsTriangle(const TriangleGeometry& tri, const Vec3& from,
                                   const Vec3& direction, val_t maxDist) {
    const Vec3 pvec = direction.cross(tri.ac);
    const val_t det = tri.ab.dot(pvec);
    if (fabsf(det) < EPSILON) return false;

    const val_t invDet = ONE / det;
    const Vec3 tvec = from - tri.a;
    const val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return false;
    const Vec3 qvec = tvec.cross(tri.ab);
    const val_t v = direction.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;
    const val_t dist = tri.ac.dot(qvec) * invDet;
    return dist > EPSILON && dist < maxDist;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const Vec3& direction,
                             val_t rayLen, idx_t i, idx_t j,
                             const TriangleGeometry* triangles, const OctreeNode* nodes,
                             const idx_t* indices) {
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 midpoint = from + d;
    const Vec3 ad{fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    // Preorder traversal uses escape indices to skip missed subtrees without a stack.
    const idx_t end = nodes[0].escape;
    for (idx_t node = 0; node < end;) {
        const OctreeNode box = nodes[node];
        if (node != 0 && !intersectsBox(box, midpoint, d, ad)) {
            node = box.escape;
            continue;
        }
        for (idx_t k = box.first; k < box.first + box.count; ++k) {
            const idx_t idx = indices[k];
            if (idx != i && idx != j &&
                intersectsTriangle(triangles[idx], from, direction, rayLen - EPSILON)) {
                return true;
            }
        }
        ++node;
    }
    return false;
}

// ============================================================================
// Deterministic Random Sampling and Form Factors
// ============================================================================

// MT19937's untempered state. The host skips draws by advancing the recurrence,
// and CUDA blocks temper only the samples they consume.
constexpr unsigned MT_SIZE = 624;
constexpr unsigned MT_SHIFT = 397;

__host__ __device__ uint32_t twistWord(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t joined = (a & 0x80000000u) | (b & 0x7fffffffu);
    return c ^ (joined >> 1) ^ ((0u - (b & 1u)) & 0x9908b0dfu);
}

struct RandomCheckpoint {
    uint32_t words[MT_SIZE];
    unsigned position;

    void seed(uint32_t value) {
        words[0] = value;
        for (unsigned i = 1; i < MT_SIZE; ++i) {
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        }
        twist();
        position = 0;
    }

    void advance(uint64_t draws) {
        draws += position;
        for (uint64_t blocks = draws / MT_SIZE; blocks > 0; --blocks) twist();
        position = draws % MT_SIZE;
    }

private:
    void twist() {
        for (unsigned i = 0; i < MT_SIZE - MT_SHIFT; ++i) {
            words[i] = twistWord(words[i], words[i + 1], words[i + MT_SHIFT]);
        }
        for (unsigned i = MT_SIZE - MT_SHIFT; i < MT_SIZE - 1; ++i) {
            words[i] = twistWord(words[i], words[i + 1], words[i - (MT_SIZE - MT_SHIFT)]);
        }
        words[MT_SIZE - 1] = twistWord(words[MT_SIZE - 1], words[0], words[MT_SHIFT - 1]);
    }
};

// For a jump of D draws, reduce x^D modulo the characteristic polynomial of
// the MT19937 recurrence. Each set coefficient k selects the state advanced
// by k words. XORing those states applies the jump without walking D draws.
constexpr unsigned MT_DEGREE = 19937;
constexpr unsigned POLY_WORDS = (MT_DEGREE + 63) / 64;
constexpr unsigned MT_POWERS[] = {
    0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681, 3908, 4135,
    4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752, 7979, 8206, 9505,
    9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374,
    11384, 11485, 11611, 11712, 11717, 11838, 11881, 11944, 11997, 12277, 12335, 12393,
    12504, 12509, 12620, 12673, 12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185,
    13190, 13243, 13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093,
    14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774, 14779, 14953,
    15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682,
    15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590, 16595,
    16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783,
    17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314, MT_DEGREE
};
static_assert(MT_DEGREE - 19314 >= 64);

// In GF(2), squaring a polynomial doubles the exponent of each term.
uint64_t squareCoefficients(uint64_t value) {
    value = (value | (value << 16)) & 0x0000ffff0000ffffULL;
    value = (value | (value << 8)) & 0x00ff00ff00ff00ffULL;
    value = (value | (value << 4)) & 0x0f0f0f0f0f0f0f0fULL;
    value = (value | (value << 2)) & 0x3333333333333333ULL;
    return (value | (value << 1)) & 0x5555555555555555ULL;
}

RandomCheckpoint checkpointAt(uint64_t draws) {
    RandomCheckpoint state;
    state.seed(42);
    // Direct advancement is cheaper for short skips.
    if (draws < (1u << 20)) {
        state.advance(draws);
        return state;
    }

    std::array<uint64_t, POLY_WORDS> poly{1};
    for (int bit = 63 - __builtin_clzll(draws); bit >= 0; --bit) {
        const unsigned shift = (draws >> bit) & 1u;
        std::array<uint64_t, 2 * POLY_WORDS + 1> squared{};
        for (unsigned w = 0; w < POLY_WORDS; ++w) {
            squared[2 * w] = squareCoefficients(static_cast<uint32_t>(poly[w])) << shift;
            squared[2 * w + 1] = squareCoefficients(poly[w] >> 32) << shift;
        }

        // The two leading nonzero terms are more than a word apart. This
        // makes it safe to cancel 64 high coefficients in one reduction.
        constexpr unsigned topBit = MT_DEGREE % 64;
        for (int w = POLY_WORDS - 1; w >= 0; --w) {
            const unsigned top = MT_DEGREE / 64 + w;
            const uint64_t high = (squared[top] >> topBit) | (squared[top + 1] << (64 - topBit));
            if (high == 0) continue;
            for (unsigned power : MT_POWERS) {
                const unsigned word = w + power / 64;
                const unsigned offset = power % 64;
                squared[word] ^= high << offset;
                if (offset != 0) squared[word + 1] ^= high >> (64 - offset);
            }
        }
        std::copy_n(squared.begin(), POLY_WORDS, poly.begin());
    }

    RandomCheckpoint result{};
    unsigned head = 0;
    for (unsigned k = 0; k < MT_DEGREE; ++k) {
        if ((poly[k / 64] >> (k % 64)) & 1u) {
            for (unsigned w = 0; w < MT_SIZE - head; ++w) result.words[w] ^= state.words[head + w];
            for (unsigned w = 0; w < head; ++w) result.words[MT_SIZE - head + w] ^= state.words[w];
        }
        const unsigned next = (head + 1) % MT_SIZE;
        state.words[head] = twistWord(state.words[head], state.words[next],
                                     state.words[(head + MT_SHIFT) % MT_SIZE]);
        head = next;
    }
    return result;
}

__device__ val_t uniformReal(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    // Match uniform_real_distribution<float>, including rounding at the upper endpoint.
    return fminf(__uint2float_rn(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 samplePoint(const TriangleGeometry& tri, uint32_t a, uint32_t b) {
    val_t u = uniformReal(a);
    val_t v = uniformReal(b);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {fmaf(tri.ac.x, v, fmaf(tri.ab.x, u, tri.a.x)),
            fmaf(tri.ac.y, v, fmaf(tri.ab.y, u, tri.a.y)),
            fmaf(tri.ac.z, v, fmaf(tri.ab.z, u, tri.a.z))};
}

__device__ val_t traceSample(const TriangleGeometry* triangles, const OctreeNode* nodes,
                             const idx_t* indices, idx_t i, idx_t j,
                             uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const TriangleGeometry& ti = triangles[i];
    const TriangleGeometry& tj = triangles[j];
    const Vec3 from = samplePoint(ti, a, b);
    const Vec3 to = samplePoint(tj, c, d);
    const Vec3 v = to - from;
    const val_t distSqr = v.squaredNorm();
    if (distSqr < EPSILON) return ZERO;
    const val_t rayLen = sqrtf(distSqr);
    const val_t cosI = v.dot(ti.normal) / rayLen;
    const val_t cosJ = (-v).dot(tj.normal) / rayLen;
    if (cosI <= ZERO || cosJ <= ZERO) return ZERO;
    if (isRayBlocked(from, to, v / rayLen, rayLen, i, j, triangles, nodes, indices)) return ZERO;
    return (cosI * cosJ) / (PI * distSqr);
}

constexpr unsigned FORM_THREADS = 128;
constexpr unsigned PAIRS_PER_STEP = FORM_THREADS / NUM_RAYS;
constexpr unsigned STEPS_PER_TILE = 16;
constexpr unsigned PAIRS_PER_TILE = PAIRS_PER_STEP * STEPS_PER_TILE;
constexpr unsigned DRAWS_PER_PAIR = NUM_RAYS * 4;
constexpr unsigned RNG_RING = 1024;
static_assert(FORM_THREADS <= MT_SIZE - MT_SHIFT);
static_assert(RNG_RING >= MT_SIZE + FORM_THREADS &&
              RNG_RING >= PAIRS_PER_STEP * DRAWS_PER_PAIR + FORM_THREADS);

struct PairTile {
    uint8_t masks[STEPS_PER_TILE];
};

__global__ void formFactors(const TriangleGeometry* triangles, const OctreeNode* nodes,
                            const idx_t* indices, const RandomCheckpoint* checkpoints,
                            const PairTile* tiles, size_t firstPair, size_t numPairs,
                            idx_t n, idx_t startRow, idx_t rows, val_t* weights,
                            uint8_t* delays, unsigned long long* nonzero) {
    __shared__ uint32_t words[RNG_RING];
    __shared__ val_t samples[FORM_THREADS];

    const unsigned lane = threadIdx.x;
    const unsigned pair = lane / NUM_RAYS;
    const unsigned ray = lane % NUM_RAYS;
    const RandomCheckpoint& checkpoint = checkpoints[blockIdx.x];
    for (unsigned k = lane; k < MT_SIZE; k += FORM_THREADS) words[k] = checkpoint.words[k];
    __syncthreads();

    unsigned consumed = checkpoint.position;
    unsigned available = MT_SIZE;
    unsigned count = 0;
    size_t local = firstPair + blockIdx.x * PAIRS_PER_TILE;
    idx_t i = local / n;
    idx_t j = local - static_cast<size_t>(i) * n + pair;
    if (j >= n) {
        j -= n;
        ++i;
    }
    for (unsigned step = 0; step < STEPS_PER_TILE; ++step) {
        if (local >= numPairs) break;
        const unsigned mask = tiles[blockIdx.x].masks[step];
        const unsigned end = consumed + __popc(mask) * DRAWS_PER_PAIR;
        for (; available < end; available += FORM_THREADS) {
            const unsigned k = available + lane;
            const unsigned old = k - MT_SIZE;
            words[k % RNG_RING] = twistWord(words[old % RNG_RING], words[(old + 1) % RNG_RING],
                                           words[(old + MT_SHIFT) % RNG_RING]);
            __syncthreads();
        }

        val_t contribution = ZERO;
        if (mask & (1u << pair)) {
            const unsigned ordinal = __popc(mask & ((1u << pair) - 1));
            const unsigned k = consumed + ordinal * DRAWS_PER_PAIR + ray * 4;
            contribution = traceSample(triangles, nodes, indices, startRow + i, j,
                                       words[k % RNG_RING], words[(k + 1) % RNG_RING],
                                       words[(k + 2) % RNG_RING], words[(k + 3) % RNG_RING]);
        }
        samples[lane] = contribution;
        __syncthreads();

        if (ray == 0 && local + pair < numPairs) {
            val_t factor = ZERO;
            for (unsigned r = 0; r < NUM_RAYS; ++r) factor += samples[lane + r];
            factor *= INV_NUM_RAYS;
            count += factor > EPSILON;
            // Adjacent receiver threads read adjacent entries during propagation.
            const size_t element = static_cast<size_t>(j) * rows + i;
            weights[element] = fminf(factor * triangles[j].area, ONE);
            delays[element] = factor > ZERO
                ? static_cast<uint8_t>(ceilf((triangles[startRow + i].center -
                                             triangles[j].center).norm() * INV_WAVE_SPEED))
                : 0;
        }
        consumed = end;
        local += PAIRS_PER_STEP;
        j += PAIRS_PER_STEP;
        if (j >= n) {
            j -= n;
            ++i;
        }
        __syncthreads();
    }

    if (nonzero) {
        // Only the first lane of each pair contributes. Combine once per warp.
        count += __shfl_xor_sync(0xffffffffu, count, NUM_RAYS);
        if (lane % 32 == 0 && count) atomicAdd(nonzero, static_cast<unsigned long long>(count));
    }
}

constexpr unsigned TILES_PER_BATCH = 1024;

struct FormBatch {
    explicit FormBatch(unsigned blocks) : checkpoints(blocks), gpuCheckpoints(blocks), gpuTiles(blocks) {}

    CudaStream stream;
    CudaBuffer<RandomCheckpoint, true> checkpoints;
    CudaBuffer<RandomCheckpoint> gpuCheckpoints;
    CudaBuffer<PairTile> gpuTiles;
};

void computeFormFactors(const std::vector<TriangleGeometry>& geometry, const Octree& octree,
                        MPI_Comm comm, int rank, idx_t startRow, idx_t rows,
                        val_t* weights, uint8_t* delays, unsigned long long* nonzero) {
    const idx_t n = geometry.size();
    const size_t pairs = static_cast<size_t>(rows) * n;
    const size_t tileCount = (pairs + PAIRS_PER_TILE - 1) / PAIRS_PER_TILE;
    // Pin the masks so copies can overlap work on the other CUDA stream.
    CudaBuffer<PairTile, true> tiles(tileCount);
    uint64_t totalActive = 0;
    const int threads = std::min<size_t>(omp_get_max_threads(), (tileCount + 63) / 64);
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:totalActive)
    for (size_t tile = 0; tile < tileCount; ++tile) {
        size_t local = tile * PAIRS_PER_TILE;
        idx_t i = startRow + local / n;
        idx_t j = local % n;
        for (auto& mask : tiles.data()[tile].masks) {
            mask = 0;
            for (unsigned pair = 0; pair < PAIRS_PER_STEP && local < pairs; ++pair, ++local) {
                if (i != j && geometry[i].normal.dot(geometry[j].normal) <= 0.99f) {
                    mask |= 1u << pair;
                }
                if (++j == n) {
                    j = 0;
                    ++i;
                }
            }
            totalActive += __builtin_popcount(mask);
        }
    }

    std::vector<OctreeNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(octree, nodes, indices);
    CudaBuffer<TriangleGeometry> gpuGeometry(n);
    CudaBuffer<OctreeNode> gpuNodes(nodes.size());
    CudaBuffer<idx_t> gpuIndices(indices.size());
    CUDA_CHECK(cudaMemcpy(gpuGeometry.data(), geometry.data(), n * sizeof(TriangleGeometry),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuNodes.data(), nodes.data(), nodes.size() * sizeof(OctreeNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuIndices.data(), indices.data(), indices.size() * sizeof(idx_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    FormBatch batches[] = {FormBatch(std::min<size_t>(tileCount, TILES_PER_BATCH)),
                           FormBatch(tileCount > TILES_PER_BATCH ? TILES_PER_BATCH : 0)};
    uint64_t preceding = 0;
    MPI_Exscan(&totalActive, &preceding, 1, MPI_UINT64_T, MPI_SUM, comm);
    if (rank == 0) preceding = 0;
    RandomCheckpoint next = checkpointAt(preceding * DRAWS_PER_PAIR);

    for (size_t first = 0, batch = 0; first < tileCount; first += TILES_PER_BATCH, ++batch) {
        FormBatch& slot = batches[batch % 2];
        if (batch >= 2) CUDA_CHECK(cudaStreamSynchronize(slot.stream));
        const unsigned blocks = std::min<size_t>(TILES_PER_BATCH, tileCount - first);
        for (unsigned k = 0; k < blocks; ++k) {
            slot.checkpoints.data()[k] = next;
            unsigned active = 0;
            for (unsigned mask : tiles.data()[first + k].masks) active += __builtin_popcount(mask);
            next.advance(active * DRAWS_PER_PAIR);
        }
        CUDA_CHECK(cudaMemcpyAsync(slot.gpuCheckpoints.data(), slot.checkpoints.data(),
                                   blocks * sizeof(RandomCheckpoint), cudaMemcpyHostToDevice, slot.stream));
        CUDA_CHECK(cudaMemcpyAsync(slot.gpuTiles.data(), tiles.data() + first,
                                   blocks * sizeof(PairTile), cudaMemcpyHostToDevice, slot.stream));
        formFactors<<<blocks, FORM_THREADS, 0, slot.stream>>>(
            gpuGeometry.data(), gpuNodes.data(), gpuIndices.data(), slot.gpuCheckpoints.data(),
            slot.gpuTiles.data(), first * PAIRS_PER_TILE, pairs, n, startRow, rows, weights, delays, nonzero);
        CUDA_CHECK(cudaGetLastError());
    }
    for (const FormBatch& slot : batches) CUDA_CHECK(cudaStreamSynchronize(slot.stream));
}

// ============================================================================
// Delayed Wave Propagation and Cross-Correlation
// ============================================================================

// Keep enough recent global frames for every possible propagation delay.
constexpr unsigned HISTORY_SLOTS = 64;
static_assert(HISTORY_SLOTS > 2 * ROOM_RADIUS * INV_WAVE_SPEED + 1);
constexpr int WAVE_THREADS = 128;

__global__ void propagate(const val_t* weights, const uint8_t* delays, const val_t* history,
                          val_t* localHistory, idx_t n, idx_t rows, idx_t firstRow, idx_t source,
                          int time, int timeOff, val_t reflectivity, bool distributed) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    // Preserve ascending emitter order and the separately rounded product.
    val_t sum = ZERO;
    if (time > 0) {
        #pragma unroll 4
        for (idx_t j = 0; j < n; ++j) {
            const size_t element = static_cast<size_t>(j) * rows + i;
            const unsigned delay = delays[element];
            if (delay == 0 || delay > static_cast<unsigned>(time)) continue;
            const size_t srcTime = distributed ? (time - delay) % HISTORY_SLOTS : time - delay;
            const val_t incoming = history[srcTime * n + j];
            if (incoming <= ZERO) continue;
            sum = __fadd_rn(sum, __fmul_rn(weights[element], incoming));
        }
    }
    const val_t emission = firstRow + i == source && time < timeOff ? ONE : ZERO;
    localHistory[static_cast<size_t>(time) * rows + i] = fmaf(reflectivity, sum, emission);
}

void runSimulation(MPI_Comm comm, int rank, const std::vector<int>& rows,
                   const std::vector<int>& offsets, idx_t n, idx_t source,
                   int timesteps, val_t reflectivity, const val_t* weights,
                   const uint8_t* delays, val_t* localHistory, val_t* sourceHistory) {
    const bool distributed = rows.size() > 1;
    const idx_t count = rows[rank];
    const idx_t first = offsets[rank];
    CudaStream stream;
    CudaBuffer<val_t> history(distributed ? static_cast<size_t>(n) * HISTORY_SLOTS : 0);
    CudaBuffer<val_t, true> exchange(distributed ? n : 0);
    std::vector<val_t> sourceValues(distributed ? timesteps : 0);

    for (int t = 0; t < timesteps; ++t) {
        propagate<<<(count + WAVE_THREADS - 1) / WAVE_THREADS, WAVE_THREADS, 0, stream>>>(
            weights, delays, distributed ? history.data() : localHistory, localHistory,
            n, count, first, source, t, timesteps / 2, reflectivity, distributed);
        CUDA_CHECK(cudaGetLastError());

        if (distributed) {
            CUDA_CHECK(cudaMemcpyAsync(exchange.data() + first, localHistory + static_cast<size_t>(t) * count,
                                       count * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange.data(), rows.data(), offsets.data(),
                           MPI_FLOAT, comm);
            sourceValues[t] = exchange.data()[source];
            if (t + 1 < timesteps) {
                CUDA_CHECK(cudaMemcpyAsync(history.data() + (t % HISTORY_SLOTS) * static_cast<size_t>(n),
                                           exchange.data(), n * sizeof(val_t), cudaMemcpyHostToDevice, stream));
            }
        }
        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == timesteps)) {
            printf("  Timestep %d/%d\n", t + 1, timesteps);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (distributed && timesteps > 0) {
        CUDA_CHECK(cudaMemcpy(sourceHistory, sourceValues.data(), timesteps * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }
}

constexpr unsigned DISTANCE_WIDTH = 32;
constexpr unsigned LAG_WORKERS = 8;
constexpr unsigned LAGS_PER_BLOCK = 64;

struct CorrelationPeak {
    val_t score;
    unsigned lag;
};

__device__ CorrelationPeak betterPeak(CorrelationPeak a, CorrelationPeak b) {
    return (b.score > a.score || (b.score == a.score && b.lag < a.lag)) ? b : a;
}

__global__ void computeDistances(const val_t* history, const val_t* sourceHistory,
                                 idx_t rows, idx_t source, unsigned timesteps, unsigned lagSpan,
                                 CorrelationPeak* partial, val_t* distances, unsigned long long* energy) {
    __shared__ CorrelationPeak peaks[LAG_WORKERS][DISTANCE_WIDTH];
    const idx_t i = blockIdx.x * DISTANCE_WIDTH + threadIdx.x;
    const unsigned worker = threadIdx.y;
    const unsigned firstLag = blockIdx.y * lagSpan;
    const unsigned endLag = min(firstLag + lagSpan, timesteps);
    CorrelationPeak best{ZERO, 0};
    bool received = false;
    if (i < rows) {
        for (unsigned lag = firstLag + worker; lag < endLag; lag += LAG_WORKERS) {
            val_t score = ZERO;
            for (unsigned time = lag; time < timesteps; ++time) {
                const size_t sourceTime = time - lag;
                const val_t a = sourceHistory ? sourceHistory[sourceTime] : history[sourceTime * rows + source];
                const val_t b = history[static_cast<size_t>(time) * rows + i];
                score = __fadd_rn(score, __fmul_rn(a, b));
                if (energy && lag == 0) received |= b > EPSILON;
            }
            best = betterPeak(best, {score, lag});
        }
    }
    peaks[worker][threadIdx.x] = best;
    __syncthreads();

    if (worker == 0) {
        if (i < rows) {
            for (unsigned w = 1; w < LAG_WORKERS; ++w) {
                best = betterPeak(best, peaks[w][threadIdx.x]);
            }
            if (partial) partial[static_cast<size_t>(blockIdx.y) * rows + i] = best;
            else distances[i] = WAVE_SPEED * static_cast<val_t>(best.lag);
        }
        if (energy && blockIdx.y == 0) {
            const unsigned count = __popc(__ballot_sync(0xffffffffu, received));
            if (threadIdx.x == 0 && count) atomicAdd(energy, static_cast<unsigned long long>(count));
        }
    }
}

__global__ void finishDistances(const CorrelationPeak* partial, idx_t rows, unsigned groups,
                                val_t* distances) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    CorrelationPeak best{ZERO, 0};
    for (size_t group = 0; group < groups; ++group) {
        best = betterPeak(best, partial[group * rows + i]);
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(best.lag);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const std::vector<val_t>& distances, size_t source,
                     const uint64_t counts[2]) {
    printf("\nValidation:\n");

    const size_t n = distances.size();
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    for (size_t i = 0; i < n; ++i) {
        val_t d = distances[i];
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);
    if (distances[source] > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", distances[source]);
    }

    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", counts[0], n);
    if (counts[0] == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           counts[1], n * n, 100.0f * counts[1] / static_cast<val_t>(n * n));
    if (counts[1] == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }
    if (!allNonNegative) return false;
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

int runBenchmark(int argc, char** argv, int worldRank, int worldSize) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (timesteps < 0) {
        if (worldRank == 0) fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const idx_t n = 20u << (2 * subdivisions);
    const idx_t source = static_cast<size_t>(sourceIdx) % n;
    const int ranks = std::min(worldSize, static_cast<int>(n));
    MPI_Comm comm = MPI_COMM_WORLD;
    if (ranks < worldSize) {
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
        if (comm == MPI_COMM_NULL) return 0;
    }
    const int rank = worldRank;
    const bool root = rank == 0;

    MPI_Comm shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) CUDA_CHECK(cudaErrorNoDevice);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::vector<int> rows(ranks);
    std::vector<int> offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = static_cast<size_t>(n) * r / ranks;
        rows[r] = static_cast<size_t>(n) * (r + 1) / ranks - offsets[r];
    }
    const size_t localPairs = static_cast<size_t>(rows[rank]) * n;

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    if (root) {
        printf("Generated icosphere mesh with %u triangles\n", n);
        printf("Building octree...\n\n");
    }
    Octree octree;
    octree.build(mesh.triangles);

    std::vector<TriangleGeometry> geometry(n);
    const int threads = std::min<size_t>(omp_get_max_threads(), (n + 255) / 256);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (idx_t i = 0; i < n; ++i) {
        const Triangle& tri = mesh.triangles[i];
        geometry[i] = {tri.a, tri.b - tri.a, tri.c - tri.a, tri.normal(), tri.center(), tri.area()};
    }

    CudaBuffer<val_t> weights(localPairs);
    CudaBuffer<uint8_t> delays(localPairs);
    CudaBuffer<unsigned long long> validation(validate ? 2 : 0);
    if (validate) {
        CUDA_CHECK(cudaMemset(validation.data(), 0, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    MPI_Barrier(comm);
    const double startPre = MPI_Wtime();
    if (root) {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");
    }
    computeFormFactors(geometry, octree, comm, rank, offsets[rank], rows[rank],
                       weights.data(), delays.data(), validate ? validation.data() + 1 : nullptr);
    MPI_Barrier(comm);
    const long preDuration = static_cast<long>((MPI_Wtime() - startPre) * 1000);
    if (root) {
        printf("  Progress: %u/%u triangles\n", n, n);
        printf("Precomputation time: %ld ms\n\n", preDuration);
        printf("Running wave propagation simulation...\n");
    }

    const double startSim = MPI_Wtime();
    CudaBuffer<val_t> localHistory(static_cast<size_t>(timesteps) * rows[rank]);
    CudaBuffer<val_t> sourceHistory(ranks > 1 ? timesteps : 0);
    runSimulation(comm, rank, rows, offsets, n, source, timesteps, reflectivity, weights.data(),
                  delays.data(), localHistory.data(), sourceHistory.data());
    MPI_Barrier(comm);
    const long simDuration = static_cast<long>((MPI_Wtime() - startSim) * 1000);
    if (root) {
        printf("Simulation time: %ld ms\n\n", simDuration);
        printf("Computing distances via cross-correlation...\n");
    }

    const double startDist = MPI_Wtime();
    const idx_t count = rows[rank];
    CudaBuffer<val_t> gpuDistances(count);
    // CUDA grid Y supports at most 65,535 blocks. Increase the lag span for longer histories.
    const unsigned lagSpan = std::max(LAGS_PER_BLOCK, (static_cast<unsigned>(timesteps) + 65534) / 65535);
    const unsigned lagGroups = std::max(1u, (timesteps + lagSpan - 1) / lagSpan);
    CudaBuffer<CorrelationPeak> partial(lagGroups > 1 ? static_cast<size_t>(lagGroups) * count : 0);
    computeDistances<<<dim3((count + DISTANCE_WIDTH - 1) / DISTANCE_WIDTH, lagGroups),
                       dim3(DISTANCE_WIDTH, LAG_WORKERS)>>>(
        localHistory.data(), sourceHistory.data(), count, source, timesteps, lagSpan, partial.data(),
        gpuDistances.data(), validation.data());
    CUDA_CHECK(cudaGetLastError());
    if (lagGroups > 1) {
        finishDistances<<<(count + WAVE_THREADS - 1) / WAVE_THREADS, WAVE_THREADS>>>(
            partial.data(), count, lagGroups, gpuDistances.data());
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<val_t> localDistances(count);
    CUDA_CHECK(cudaMemcpy(localDistances.data(), gpuDistances.data(), count * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    std::vector<val_t> distances(root ? n : 0);
    MPI_Gatherv(localDistances.data(), count, MPI_FLOAT, distances.data(), rows.data(), offsets.data(),
                MPI_FLOAT, 0, comm);

    uint64_t globalCounts[2] = {};
    if (validate) {
        unsigned long long deviceCounts[2];
        CUDA_CHECK(cudaMemcpy(deviceCounts, validation.data(), sizeof(deviceCounts), cudaMemcpyDeviceToHost));
        const uint64_t counts[] = {deviceCounts[0], deviceCounts[1]};
        MPI_Reduce(counts, globalCounts, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
    }

    const double localElapsed = MPI_Wtime() - startPre;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int result = 0;
    if (root) {
        const long distDuration = static_cast<long>((MPI_Wtime() - startDist) * 1000);
        const long totalTime = static_cast<long>(maxElapsed * 1000);
        const double pairs = static_cast<double>(n) * n;
        const size_t matrixBytes = static_cast<size_t>(n) * n * (sizeof(val_t) + sizeof(uint8_t));
        const size_t historyEntries = static_cast<size_t>(n) * timesteps +
                                      (ranks > 1 ? ranks * (n * HISTORY_SLOTS + static_cast<size_t>(timesteps)) : 0);
        const size_t totalBytes = matrixBytes + historyEntries * sizeof(val_t);

        printf("Distance computation time: %ld ms\n\n", distDuration);
        printf("Total computation time: %ld ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %u\n", n);
        printf("  Timesteps: %d\n", timesteps);
        printf("  Form factor computations: %.2e\n", pairs);
        printf("  Simulation operations: %.2e\n", pairs * timesteps);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * timesteps * timesteps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
        printf("  Memory usage: %.2f MB\n", totalBytes / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", computeHash(distances));

        if (printResults) {
            std::vector<double> distData(distances.begin(), distances.end());
            print_results(distData, "Distances");
        }
        if (validate && !validateResults(distances, source, globalCounts)) result = 1;
    }
    if (validate) MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
    return result;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int result = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
