/**
 * Room Response Simulation Benchmark
 * 
 * This benchmark distributes receiver triangles among MPI ranks for room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

void checkCuda(cudaError_t error, const char* operation, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA failure at line %d (%s): %s\n", line, operation,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation, __LINE__)

template <typename T>
class DeviceBuffer {
    T* ptr = nullptr;
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t count) {
        if (count) CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
};

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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

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

        // Large independent root octants can be built concurrently.
        #pragma omp parallel for if(indices.size() == allTriangles->size() && indices.size() >= 1024) num_threads(std::min(8, omp_get_max_threads()))
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

constexpr int MT_WORDS = 624;
__host__ __device__ uint32_t twistWord(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return c ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
}

// Keep the serial MT19937 stream position in draw units. Each rank can seek
// directly to its first draw even when preceding ranks own billions of draws.
class RandomGenerator {
    static constexpr int words = MT_WORDS;
    static constexpr int lag = 397;
    static constexpr int degree = 19937;
    using State = std::array<uint32_t, words>;
    State state{};
    int cursor = 0;

    void twist() {
        for (int i = 0; i < words - lag; ++i)
            state[i] = twistWord(state[i], state[i + 1], state[i + lag]);
        for (int i = words - lag; i < words - 1; ++i)
            state[i] = twistWord(state[i], state[i + 1], state[i + lag - words]);
        state.back() = twistWord(state.back(), state.front(), state[lag - 1]);
    }

    // Compute x^offset modulo the characteristic polynomial of the one-word
    // MT19937 transition over GF(2), using binary exponentiation. The listed
    // exponents are its nonzero coefficients below the leading x^19937.
    static constexpr int coefficients[] = {
        0,1189,1416,1585,1643,1870,2493,2773,3000,3227,3454,3681,3908,
        4135,4362,4753,5661,6337,6569,7129,7477,7525,7583,7752,7979,
        8206,9505,9901,9969,10128,10693,10761,10920,11089,11147,11157,
        11215,11321,11374,11384,11485,11611,11712,11717,11838,11881,
        11944,11997,12277,12335,12393,12504,12509,12620,12673,12731,
        12736,12789,12905,12958,12963,13137,13185,13190,13243,13301,
        13412,13528,13533,13639,13697,13760,13813,13866,14093,14151,
        14209,14320,14325,14436,14547,14552,14605,14721,14774,14779,
        14953,15001,15006,15059,15117,15228,15344,15349,15455,15513,
        15576,15629,15682,15909,15967,16025,16136,16141,16252,16363,
        16368,16421,16537,16590,16595,16817,16822,16875,16933,17044,
        17160,17271,17329,17445,17498,17725,17783,17841,17952,18068,
        18179,18237,18406,18633,18691,18860,19087,19314
    };
    using Polynomial = std::array<uint64_t, (degree + 63) / 64>;

    static Polynomial jumpPolynomial(uint64_t offset) {
        Polynomial poly{};
        poly[0] = 1;
        for (int bit = 63 - __builtin_clzll(offset); bit >= 0; --bit) {
            std::array<uint64_t, 2 * Polynomial{}.size() + 1> square{};
            int extra = (offset >> bit) & 1U;
            for (int i = 0; i < degree; ++i) {
                if ((poly[i / 64] >> (i % 64)) & 1U) {
                    int exponent = 2 * i + extra;
                    square[exponent / 64] |= uint64_t{1} << (exponent % 64);
                }
            }
            // Reduce 64 consecutive high coefficients together. The second
            // highest term lies 623 positions below the leading coefficient.
            for (int base = ((degree - 1) / 64) * 64; base >= 0; base -= 64) {
                int word = (degree + base) / 64;
                constexpr int shift = degree % 64;
                uint64_t high = (square[word] >> shift) |
                                (square[word + 1] << (64 - shift));
                if (!high) continue;
                for (int term : coefficients) {
                    int position = (base + term) / 64;
                    int bits = term % 64;
                    square[position] ^= high << bits;
                    if (bits) square[position + 1] ^= high >> (64 - bits);
                }
            }
            std::copy_n(square.begin(), poly.size(), poly.begin());
            poly.back() &= (uint64_t{1} << (degree % 64)) - 1;
        }
        return poly;
    }

    void jump(uint64_t offset) {
        Polynomial poly = jumpPolynomial(offset);
        // Duplicate the circular state so every polynomial term can be XORed
        // in logical stream order with a single contiguous vector loop.
        std::array<uint32_t, 2 * words> ring;
        std::copy(state.begin(), state.end(), ring.begin());
        std::copy(state.begin(), state.end(), ring.begin() + words);
        state.fill(0);
        int head = 0;
        for (int i = 0; i < degree; ++i) {
            if ((poly[i / 64] >> (i % 64)) & 1U) {
                for (int j = 0; j < words; ++j) state[j] ^= ring[head + j];
            }
            uint32_t next = twistWord(ring[head], ring[head + 1], ring[head + lag]);
            ring[head] = ring[head + words] = next;
            if (++head == words) head = 0;
        }
    }

public:
    explicit RandomGenerator(uint64_t firstDraw) {
        state.front() = 42U;
        for (int i = 1; i < words; ++i)
            state[i] = 1812433253U * (state[i-1] ^ (state[i-1] >> 30)) + i;
        twist();
        if (firstDraw > (1U << 20)) {
            jump(firstDraw);
        } else {
            for (uint64_t i = 0; i < firstDraw / words; ++i) twist();
            cursor = firstDraw % words;
        }
    }

    // A CUDA block expands this 624-word window into its own segment of the
    // original stream. Successive calls may straddle arbitrary batch edges.
    void checkpoint(uint32_t* output, size_t skip) {
        int remaining = words - cursor;
        std::copy_n(state.data() + cursor, remaining, output);
        if (cursor != 0) {
            twist();
            std::copy_n(state.data(), cursor, output + remaining);
            if (skip < static_cast<size_t>(remaining)) {
                std::copy_n(output, words, state.data());
                cursor = static_cast<int>(skip);
                return;
            }
            skip -= remaining;
        }
        for (size_t i = 0; i < skip / words; ++i) twist();
        cursor = skip % words;
    }

};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return CUDART_INF_F;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return CUDART_INF_F;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return CUDART_INF_F;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// A preorder layout lets a GPU ray skip a missed subtree without a stack.
struct FlatNode {
    Vec3 center, halfExtent;
    int end, firstTriangle, count;
};

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<int>& indices) {
    const size_t start = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flatten(*child, nodes, indices);
    }
    nodes[start].end = static_cast<int>(nodes.size());
}

__device__ bool intersectsBox(const FlatNode& node, Vec3 midpoint, Vec3 d, Vec3 ad) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    return fabsf(c.x) <= h.x + ad.x && fabsf(c.y) <= h.y + ad.y &&
           fabsf(c.z) <= h.z + ad.z &&
           fabsf(d.y * c.z - d.z * c.y) <= h.y * ad.z + h.z * ad.y + EPSILON &&
           fabsf(d.z * c.x - d.x * c.z) <= h.z * ad.x + h.x * ad.z + EPSILON &&
           fabsf(d.x * c.y - d.y * c.x) <= h.x * ad.y + h.y * ad.x + EPSILON;
}

__device__ bool isRayBlocked(Vec3 from, Vec3 dir, float len, int i, int j,
                             const Triangle* triangles, const FlatNode* nodes, const int* indices) {
    const Vec3 norm = dir / len;
    const Vec3 d = dir * 0.5f;
    const Vec3 midpoint = from + d;
    const Vec3 ad{fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const int end = nodes[0].end;
    for (int k = 0; k < end; ) {
        const FlatNode node = nodes[k];
        if (k && !intersectsBox(node, midpoint, d, ad)) {
            k = node.end;
            continue;
        }
        for (int q = 0; q < node.count; ++q) {
            const int tri = indices[node.firstTriangle + q];
            if (tri == i || tri == j) continue;
            const Triangle& t = triangles[tri];
            const float dist = rayTriangleIntersect(from, norm, t.a, t.b, t.c);
            if (dist > EPSILON && dist < len - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

__device__ Vec3 pointOnTriangle(const Triangle& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
}

// Each bit represents one source in a row. Only set bits consume RNG draws.
struct PairGroup {
    unsigned long long firstSample;
    uint32_t mask;
};

// Expand sparse checkpoints on the accelerator. MT19937's next 624 words
// depend on the previous window and on new words 227 positions behind.
constexpr int CHECKPOINT_DRAWS = MT_WORDS * 12;

__global__ void expandCheckpoints(const uint32_t* checkpoints, size_t draws, float* samples) {
    __shared__ uint32_t windows[2][MT_WORDS];
    const int tid = threadIdx.x;
    const size_t start = size_t(blockIdx.x) * CHECKPOINT_DRAWS;
    const int count = min(size_t(CHECKPOINT_DRAWS), draws - start);
    uint32_t* current = windows[0];
    uint32_t* next = windows[1];
    for (int k = tid; k < MT_WORDS; k += blockDim.x)
        current[k] = checkpoints[size_t(blockIdx.x) * MT_WORDS + k];
    __syncthreads();
    for (int base = 0; base < count; base += MT_WORDS) {
        for (int k = tid; k < MT_WORDS && base + k < count; k += blockDim.x) {
            uint32_t y = current[k];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            samples[start + base + k] = fminf(__uint2float_rn(y) * 0x1p-32f, 0x1.fffffep-1f);
        }
        if (base + MT_WORDS >= count) break;
        // Write a separate window so simultaneous readers always see the old state.
        if (tid < 227) next[tid] = twistWord(current[tid], current[tid + 1], current[tid + 397]);
        __syncthreads();
        if (tid < 227) next[tid + 227] = twistWord(current[tid + 227], current[tid + 228], next[tid]);
        __syncthreads();
        if (tid < 169) next[tid + 454] = twistWord(current[tid + 454], current[tid + 455], next[tid + 227]);
        if (tid == 169) next[623] = twistWord(current[623], next[0], next[396]);
        __syncthreads();
        uint32_t* old = current;
        current = next;
        next = old;
    }
}

constexpr int FORM_THREADS = 128;
__global__ void computePairBatch(size_t firstGroup, size_t groupCount, int groupsPerRow,
                                 int rows, int firstRow, int n, const PairGroup* groups,
                                 unsigned long long sampleBase, const float4* samples,
                                 const Triangle* triangles, const FlatNode* nodes,
                                 const int* indices, const float* areas, float* weights,
                                 unsigned char* delays, unsigned long long* nonZero) {
    __shared__ float contributions[FORM_THREADS];
    const int ray = threadIdx.x % NUM_RAYS;
    const unsigned pair = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_RAYS;
    if (pair / 32 >= groupCount) return;
    const size_t groupId = firstGroup + pair / 32;
    const int local = groupId / groupsPerRow;
    const int i = firstRow + local;
    const int bit = pair % 32;
    const int j = (groupId % groupsPerRow) * 32 + bit;
    if (j >= n) return;
    const PairGroup group = groups[groupId];
    const uint32_t flag = 1u << bit;
    float contribution = 0.0f;
    if (group.mask & flag) {
        const size_t sample = group.firstSample - sampleBase + __popc(group.mask & (flag - 1));
        const float4 r = samples[sample * NUM_RAYS + ray];
        const Vec3 from = pointOnTriangle(triangles[i], r.x, r.y);
        const Vec3 to = pointOnTriangle(triangles[j], r.z, r.w);
        const Vec3 dir = to - from;
        const float sqr = dir.squaredNorm();
        const float len = sqrtf(sqr);
        if (sqr >= EPSILON) {
            const float ci = dir.dot(triangles[i].normal()) / len;
            const float cj = (-dir).dot(triangles[j].normal()) / len;
            if (ci > ZERO && cj > ZERO && !isRayBlocked(from, dir, len, i, j, triangles, nodes, indices))
                contribution = ci * cj / (PI * sqr);
        }
    }
    contributions[threadIdx.x] = contribution;
    // A half-warp shares one triangle pair. Keep the serial ray sum order.
    const unsigned mask = (threadIdx.x & NUM_RAYS) ? 0xffff0000u : 0x0000ffffu;
    __syncwarp(mask);
    if (ray == 0) {
        float kij = ZERO;
        for (int k = 0; k < NUM_RAYS; ++k) kij += contributions[threadIdx.x + k];
        kij *= INV_NUM_RAYS;
        const size_t dst = static_cast<size_t>(j) * rows + local;
        weights[dst] = fminf(kij * areas[j], ONE);
        delays[dst] = static_cast<unsigned char>(
            ceilf((triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
        if (nonZero && kij > EPSILON) atomicAdd(nonZero, 1ULL);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    int rank = 0, ranks = 1;
    int first = 0, rows = 0;
    std::vector<int> counts, offsets;
    size_t numTriangles, numTimesteps, sourceIndex;
    val_t reflectivity;
    bool validate;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, distances;
    Octree octree;
    DeviceBuffer<val_t> weights;
    DeviceBuffer<unsigned char> delays;
    DeviceBuffer<val_t> history, sourceSignal;
    unsigned long long nonZeroKij = 0;
    int receivedEnergy = 0;
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", state.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    int n = static_cast<int>(state.numTriangles);
    state.counts.resize(state.ranks);
    state.offsets.resize(state.ranks);
    for (int r = 0; r < state.ranks; ++r) {
        state.counts[r] = n / state.ranks + (r < n % state.ranks);
        state.offsets[r] = r * (n / state.ranks) + std::min(r, n % state.ranks);
    }
    state.first = state.offsets[state.rank];
    state.rows = state.counts[state.rank];
    if (state.rank == 0) {
        printf("Generated icosphere mesh with %d triangles\nBuilding octree...\n", n);
    }
    state.octree.build(state.triangles);
    state.areas.resize(n);
    const int areaThreads = std::min(omp_get_max_threads(), (n + 255) / 256);
    #pragma omp parallel for schedule(static) num_threads(areaThreads)
    for (int i = 0; i < n; ++i) state.areas[i] = state.triangles[i].area();
    state.weights.allocate(state.rows * state.numTriangles);
    state.delays.allocate(state.rows * state.numTriangles);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const int n = static_cast<int>(state.numTriangles);
    const int groupsPerRow = (n + 31) / 32;
    const size_t groupCount = static_cast<size_t>(state.rows) * groupsPerRow;
    std::vector<PairGroup> groups(groupCount);
    std::vector<unsigned long long> rowCounts(state.rows + 1, 0);
    const size_t pairCount = size_t(state.rows) * n;
    const int threads = std::min<size_t>(omp_get_max_threads(), 1 + pairCount / 65536);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int local = 0; local < state.rows; ++local) {
        const int i = state.first + local;
        const Vec3 normal = state.triangles[i].normal();
        unsigned long long count = 0;
        for (int g = 0; g < groupsPerRow; ++g) {
            uint32_t mask = 0;
            const int start = g * 32;
            const int width = std::min(32, n - start);
            for (int k = 0; k < width; ++k) {
                const int j = start + k;
                if (i != j && normal.dot(state.triangles[j].normal()) <= 0.99f)
                    mask |= 1u << k;
            }
            groups[static_cast<size_t>(local) * groupsPerRow + g] = {count, mask};
            count += __builtin_popcount(mask);
        }
        rowCounts[local + 1] = count;
    }
    for (int i = 1; i <= state.rows; ++i) rowCounts[i] += rowCounts[i - 1];
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int local = 0; local < state.rows; ++local) {
        for (int g = 0; g < groupsPerRow; ++g)
            groups[static_cast<size_t>(local) * groupsPerRow + g].firstSample += rowCounts[local];
    }
    unsigned long long firstSample = 0;
    MPI_Exscan(&rowCounts.back(), &firstSample, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (state.rank == 0) firstSample = 0;  // MPI_Exscan leaves the first result undefined.
    if (!state.rows) return;
    RandomGenerator rng(firstSample * (4 * NUM_RAYS));

    std::vector<FlatNode> nodes;
    std::vector<int> indices;
    flatten(state.octree, nodes, indices);
    DeviceBuffer<FlatNode> gpuNodes(nodes.size());
    DeviceBuffer<int> gpuIndices(indices.size());
    DeviceBuffer<Triangle> gpuTriangles(n);
    DeviceBuffer<float> gpuAreas(n);
    DeviceBuffer<PairGroup> gpuGroups(groupCount);
    CUDA_CHECK(cudaMemcpy(gpuNodes.data(), nodes.data(), nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuIndices.data(), indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuTriangles.data(), state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuAreas.data(), state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuGroups.data(), groups.data(), groupCount * sizeof(PairGroup), cudaMemcpyHostToDevice));

    DeviceBuffer<unsigned long long> nonZero(state.validate ? 1 : 0);
    if (state.validate) CUDA_CHECK(cudaMemset(nonZero.data(), 0, sizeof(unsigned long long)));

    // While the GPU traces a batch, fill the pinned input for the next one.
    constexpr size_t groupsPerBatch = 512;
    const size_t capacity = std::min(groupCount, groupsPerBatch) * 32 * NUM_RAYS * 4;
    DeviceBuffer<float> gpuSamples(capacity);
    const size_t checkpointCapacity = ((capacity + CHECKPOINT_DRAWS - 1) / CHECKPOINT_DRAWS) * MT_WORDS;
    DeviceBuffer<uint32_t> gpuCheckpoints(checkpointCapacity);
    uint32_t* hostCheckpoints;
    CUDA_CHECK(cudaMallocHost(&hostCheckpoints, checkpointCapacity * sizeof(uint32_t)));
    cudaEvent_t copied;
    CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    for (size_t first = 0; first < groupCount; first += groupsPerBatch) {
        const size_t last = std::min(groupCount, first + groupsPerBatch);
        const auto base = groups[first].firstSample;
        const auto end = last == groupCount ? rowCounts.back() : groups[last].firstSample;
        const size_t draws = static_cast<size_t>(end - base) * NUM_RAYS * 4;
        if (first) CUDA_CHECK(cudaEventSynchronize(copied));
        const int chunks = (draws + CHECKPOINT_DRAWS - 1) / CHECKPOINT_DRAWS;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            size_t length = std::min<size_t>(CHECKPOINT_DRAWS, draws - size_t(chunk) * CHECKPOINT_DRAWS);
            rng.checkpoint(hostCheckpoints + size_t(chunk) * MT_WORDS, length);
        }
        CUDA_CHECK(cudaMemcpyAsync(gpuCheckpoints.data(), hostCheckpoints,
                                   size_t(chunks) * MT_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(copied));
        if (chunks) {
            expandCheckpoints<<<chunks, 256>>>(gpuCheckpoints.data(), draws, gpuSamples.data());
            CUDA_CHECK(cudaGetLastError());
        }
        const int blocks = static_cast<int>((last - first) * 32 * NUM_RAYS / FORM_THREADS);
        computePairBatch<<<blocks, FORM_THREADS>>>(
            first, last - first, groupsPerRow, state.rows, state.first, n, gpuGroups.data(), base,
            reinterpret_cast<const float4*>(gpuSamples.data()), gpuTriangles.data(), gpuNodes.data(),
            gpuIndices.data(), gpuAreas.data(), state.weights.data(), state.delays.data(), nonZero.data());
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (state.validate)
        CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, nonZero.data(), sizeof(state.nonZeroKij), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(copied));
    CUDA_CHECK(cudaFreeHost(hostCheckpoints));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// A radius-10 mesh has at most 41 steps of propagation delay (including
// roundoff). Keep a power-of-two ring for remote histories on each GPU.
constexpr int HISTORY_SLOTS = 64;
constexpr int MAX_TIME_BATCH = 16;
static_assert(HISTORY_SLOTS > 2 * 10 * INV_WAVE_SPEED + 1);

__global__ void findMinimumDelay(const unsigned char* delays, const float* weights,
                                 size_t pairs, int* minimum) {
    int best = MAX_TIME_BATCH;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs;
         p += gridDim.x * blockDim.x) {
        if (weights[p] > ZERO) best = min(best, int(delays[p]));
    }
    for (int step = 16; step; step /= 2)
        best = min(best, __shfl_down_sync(0xffffffff, best, step));
    if (threadIdx.x % 32 == 0 && best < MAX_TIME_BATCH) atomicMin(minimum, best);
}

// Each warp owns four receivers. Its eight groups fetch independent emitters,
// then shuffle their contributions into mesh order for the original float sum.
template<bool useRing>
__global__ void propagate(const float* __restrict__ weights, const unsigned char* delays,
                          const float* previous, float* history, int first, int rows,
                          int n, int t0, int emissionEnd, int source, float rho) {
    constexpr int receivers = 4;
    constexpr int emitters = 32 / receivers;
    int x = threadIdx.x % receivers;
    int laneGroup = threadIdx.x / receivers;
    int row = blockIdx.x * receivers + x;
    int t = t0 + blockIdx.y;
    float sum = ZERO;
    for (int j0 = 0; j0 < n; j0 += emitters) {
        int j = j0 + laneGroup;
        float contribution = ZERO;
        if (j < n && row < rows) {
            size_t pair = size_t(j) * rows + row;
            int delay = delays[pair];
            float w = weights[pair];
            if (delay <= t && w > ZERO) {
                int slot = useRing ? ((t - delay) & (HISTORY_SLOTS - 1)) : t - delay;
                float incoming = previous[size_t(slot) * n + j];
                if (!(incoming <= ZERO)) contribution = w * incoming;
            }
        }
        #pragma unroll
        for (int group = 0; group < emitters; ++group)
            sum += __shfl_sync(0xffffffff, contribution, group * receivers + x);
    }
    if (row < rows && laneGroup == 0) {
        float emission = first + row == source && t < emissionEnd ? ONE : ZERO;
        history[size_t(t) * rows + row] = rho * sum + emission;
    }
}

// MPI concatenates each rank's time-major batch. Transpose just the rank/time
// dimensions into time-major global snapshots for the next propagation batch.
__global__ void unpackSnapshots(const float* packed, const int2* owners, float* ring,
                                float* sourceSignal, int source, int n, int t0, int batch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int t = t0 + blockIdx.y;
    int2 part = owners[j]; // first global triangle, local count
    float value = packed[size_t(part.x) * batch + blockIdx.y * part.y + j - part.x];
    ring[size_t(t & (HISTORY_SLOTS - 1)) * n + j] = value;
    if (j == source) sourceSignal[t] = value;
}

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    int n = int(state.numTriangles);
    int rows = state.rows;
    int timesteps = int(state.numTimesteps);
    state.history.allocate(size_t(timesteps) * rows);
    if (timesteps == 0) return;

    // Every positive form factor couples distinct triangle centers, so its
    // delay is positive. Timesteps within a minimum-delay window are independent.
    size_t pairs = size_t(rows) * n;
    int batchLimit = MAX_TIME_BATCH;
    if (pairs) {
        DeviceBuffer<int> minimum(1);
        CUDA_CHECK(cudaMemcpy(minimum.data(), &batchLimit, sizeof(int), cudaMemcpyHostToDevice));
        int blocks = int(std::min<size_t>((pairs + 255) / 256, 1024));
        findMinimumDelay<<<blocks, 256>>>(state.delays.data(), state.weights.data(), pairs,
                                           minimum.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&batchLimit, minimum.data(), sizeof(int), cudaMemcpyDeviceToHost));
    }

    bool distributed = state.ranks > 1;
    DeviceBuffer<float> ring, packed;
    DeviceBuffer<int2> deviceOwners;
    float* exchange = nullptr;
    std::vector<int> counts(state.ranks), offsets(state.ranks);
    if (distributed) {
        MPI_Allreduce(MPI_IN_PLACE, &batchLimit, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        batchLimit = std::min(timesteps, batchLimit);
        size_t capacity = size_t(n) * batchLimit;
        ring.allocate(size_t(n) * std::min(timesteps, HISTORY_SLOTS));
        packed.allocate(capacity);
        state.sourceSignal.allocate(timesteps);
        CUDA_CHECK(cudaMallocHost(&exchange, capacity * sizeof(float)));
        std::vector<int2> owners(n);
        for (int r = 0; r < state.ranks; ++r) {
            int first = state.offsets[r], count = state.counts[r];
            std::fill_n(owners.begin() + first, count, make_int2(first, count));
        }
        deviceOwners.allocate(n);
        CUDA_CHECK(cudaMemcpy(deviceOwners.data(), owners.data(), n * sizeof(int2),
                              cudaMemcpyHostToDevice));
    }

    for (int t = 0; t < timesteps;) {
        int batch = std::min(batchLimit, timesteps - t);
        if (rows) {
            dim3 grid((rows + 3) / 4, batch);
            if (distributed) {
                propagate<true><<<grid, 32>>>(state.weights.data(), state.delays.data(),
                    ring.data(), state.history.data(), state.first, rows, n, t,
                    timesteps / 2, state.sourceIndex, state.reflectivity);
            } else {
                propagate<false><<<grid, 32>>>(state.weights.data(), state.delays.data(),
                    state.history.data(), state.history.data(), state.first, rows, n, t,
                    timesteps / 2, state.sourceIndex, state.reflectivity);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (distributed) {
            for (int r = 0; r < state.ranks; ++r) {
                counts[r] = batch * state.counts[r];
                offsets[r] = batch * state.offsets[r];
            }
            if (rows) {
                CUDA_CHECK(cudaMemcpy(exchange + offsets[state.rank],
                    state.history.data() + size_t(t) * rows, size_t(batch) * rows * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange, counts.data(), offsets.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(packed.data(), exchange, size_t(batch) * n * sizeof(float),
                                  cudaMemcpyHostToDevice));
            dim3 grid((n + 127) / 128, batch);
            unpackSnapshots<<<grid, 128>>>(packed.data(), deviceOwners.data(), ring.data(),
                state.sourceSignal.data(), state.sourceIndex, n, t, batch);
            CUDA_CHECK(cudaGetLastError());
        }
        t += batch;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (exchange) CUDA_CHECK(cudaFreeHost(exchange));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

constexpr int LAGS_PER_TILE = 32;

__global__ void correlate(const float* history, const float* signal, int signalStride,
                          int rows, int timesteps, float* distances, float* maxima,
                          int* lags, int* receivedEnergy) {
    constexpr int lagGroups = 8;
    __shared__ float bestValues[lagGroups][32];
    __shared__ int bestLags[lagGroups][32];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * 32 + x;
    float best = ZERO;
    int bestLag = 0;
    bool energized = false;
    int tile = blockIdx.x;
    int firstLag = tile * LAGS_PER_TILE;
    int lastLag = gridDim.x == 1 ? timesteps : firstLag + min(timesteps - firstLag, LAGS_PER_TILE);
    if (row < rows) {
        for (int lag = firstLag + y; lag < lastLag; lag += lagGroups) {
            float sum = ZERO;
            for (int k = 0; k < timesteps - lag; ++k) {
                float value = history[size_t(lag + k) * rows + row];
                sum += signal[size_t(k) * signalStride] * value;
                if (receivedEnergy && lag == 0 && value > EPSILON) energized = true;
            }
            if (sum > best) {
                best = sum;
                bestLag = lag;
            }
        }
    }
    bestValues[y][x] = best;
    bestLags[y][x] = bestLag;
    __syncthreads();
    if (y == 0) {
        if (receivedEnergy && tile == 0) {
            int count = __popc(__ballot_sync(0xffffffff, energized));
            if (x == 0 && count) atomicAdd(receivedEnergy, count);
        }
        if (row < rows) {
            for (int g = 1; g < lagGroups; ++g) {
                float other = bestValues[g][x];
                int lag = bestLags[g][x];
                if (other > best || (other == best && lag < bestLag)) {
                    best = other;
                    bestLag = lag;
                }
            }
            if (maxima) {
                size_t index = size_t(tile) * rows + row;
                maxima[index] = best;
                lags[index] = bestLag;
            } else {
                distances[row] = WAVE_SPEED * float(bestLag);
            }
        }
    }
}

__global__ void selectDistances(const float* maxima, const int* lags, float* distances,
                                int rows, int tiles) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float best = ZERO;
    int lag = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        size_t index = size_t(tile) * rows + row;
        if (maxima[index] > best) {
            best = maxima[index];
            lag = lags[index];
        }
    }
    distances[row] = WAVE_SPEED * float(lag);
}

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    DeviceBuffer<float> distances(state.rows);
    DeviceBuffer<int> energy(state.validate ? 1 : 0);
    std::vector<float> local(state.rows);
    int received = 0;
    if (state.rows && state.numTimesteps) {
        const float* signal = state.ranks > 1 ? state.sourceSignal.data()
            : state.history.data() + state.sourceIndex;
        int signalStride = state.ranks > 1 ? 1 : state.rows;
        if (state.validate) CUDA_CHECK(cudaMemset(energy.data(), 0, sizeof(int)));
        int tiles = state.numTimesteps <= 2 * LAGS_PER_TILE ? 1
            : int((state.numTimesteps + LAGS_PER_TILE - 1) / LAGS_PER_TILE);
        size_t partials = tiles == 1 ? 0 : size_t(tiles) * state.rows;
        DeviceBuffer<float> maxima(partials);
        DeviceBuffer<int> lags(partials);
        dim3 grid(tiles, (state.rows + 31) / 32);
        correlate<<<grid, dim3(32, 8)>>>(state.history.data(), signal, signalStride,
            state.rows, state.numTimesteps, distances.data(), maxima.data(), lags.data(), energy.data());
        CUDA_CHECK(cudaGetLastError());
        if (tiles > 1) {
            selectDistances<<<(state.rows + 127) / 128, 128>>>(maxima.data(), lags.data(),
                distances.data(), state.rows, tiles);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(local.data(), distances.data(), state.rows * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (state.validate)
            CUDA_CHECK(cudaMemcpy(&received, energy.data(), sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (state.validate)
        MPI_Reduce(&received, &state.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (state.rank == 0) state.distances.resize(state.numTriangles);
    MPI_Gatherv(local.data(), state.rows, MPI_FLOAT, state.distances.data(),
                state.counts.data(), state.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    int receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    auto nonZeroKij = state.nonZeroKij;
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
        uint32_t bits;
        memcpy(&bits, &state.distances[i], sizeof(bits));
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

int benchmark(int argc, char** argv) {
    SimulationState state;
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.ranks);
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
            if (state.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (state.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (state.rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    if (timesteps < 0) {
        if (state.rank == 0) fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
    }
    state.validate = validate;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    computeFormFactors(state);
    CUDA_CHECK(cudaDeviceSynchronize());
    double preEnd = MPI_Wtime();
    runSimulation(state);
    CUDA_CHECK(cudaDeviceSynchronize());
    double simEnd = MPI_Wtime();
    computeDistances(state);
    double distEnd = MPI_Wtime();

    if (validate) {
        unsigned long long total;
        MPI_Reduce(&state.nonZeroKij, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (state.rank == 0) state.nonZeroKij = total;
    }
    const double elapsed[] = {preEnd - start, simEnd - preEnd, distEnd - simEnd, distEnd - start};
    double longest[4];
    MPI_Reduce(elapsed, longest, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (state.rank != 0) return 0;

    printf("Precomputation time: %.3f ms\n", 1000 * longest[0]);
    printf("Simulation time: %.3f ms\n", 1000 * longest[1]);
    printf("Distance computation time: %.3f ms\n", 1000 * longest[2]);
    double totalTime = 1000 * longest[3];
    printf("Total computation time: %.3f ms\n", totalTime);

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

    size_t matrixBytes = n * n * (sizeof(val_t) + sizeof(unsigned char));
    size_t historyBytes = t * (n + (state.ranks > 1 ? size_t(state.ranks) : 0)) * sizeof(val_t);
    printf("  Persistent simulation arrays (all GPUs): %.2f MB\n",
           (matrixBytes + historyBytes) / (1024.0 * 1024.0));

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

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result;
    try {
        result = benchmark(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Simulation failed: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
