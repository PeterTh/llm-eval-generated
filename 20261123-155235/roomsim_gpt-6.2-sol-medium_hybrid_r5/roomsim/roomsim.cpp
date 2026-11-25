/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI, OpenMP, and CUDA benchmark for room impulse response
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
#include <climits>
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
constexpr val_t ROOM_RADIUS = 10.0f;
// A diameter takes 40 timesteps, with one extra slot for rounding in ceil.
constexpr int MAX_DELAY = static_cast<int>(2 * ROOM_RADIUS * INV_WAVE_SPEED) + 1;

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

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
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

        #pragma omp parallel if(triangles.size() >= 320)
        #pragma omp single
        buildNode(allIndices, minBound, maxBound, 0);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax, int depth) {
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
                Octree* child = children[i].get();
                if (depth < 2 && childIndices[i].size() >= 64) {
                    #pragma omp task firstprivate(child, i, childMin, childMax, depth) shared(childIndices)
                    child->buildNode(childIndices[i], childMin, childMax, depth + 1);
                } else {
                    child->buildNode(childIndices[i], childMin, childMax, depth + 1);
                }
            }
        }
        #pragma omp taskwait
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
        #pragma omp parallel for schedule(static) if(faces.size() >= 320)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            // Reverse winding to make normals point inward
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T, bool Pinned = false>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count) {
        if (count == 0) return;
        if constexpr (Pinned) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
        }
    }

    ~CudaBuffer() {
        if (!data) return;
        if constexpr (Pinned) cudaFreeHost(data);
        else cudaFree(data);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* data = nullptr;
};

struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    ~MpiSession() { MPI_Finalize(); }

    void selectDevice() const {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);
        int devices;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) CUDA_CHECK(cudaErrorNoDevice);
        CUDA_CHECK(cudaSetDevice(localRank % devices));
    }

    int rank, size;
};

struct RowPartition {
    RowPartition(int n, const MpiSession& mpi)
        : counts(mpi.size), offsets(mpi.size) {
        for (int r = 0; r < mpi.size; ++r) {
            counts[r] = n / mpi.size + (r < n % mpi.size);
            if (r > 0) offsets[r] = offsets[r - 1] + counts[r - 1];
        }
        first = offsets[mpi.rank];
        count = counts[mpi.rank];
    }

    std::vector<int> counts, offsets;
    int first, count;
};

// Flatten the octree in traversal order. Each escape index skips a complete
// subtree, so CUDA threads can traverse it without a stack.
struct DeviceNode {
    Vec3 center, halfExtent;
    idx_t escape, firstTriangle, triangleCount;
};

void flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes,
                   std::vector<idx_t>& indices) {
    idx_t current = static_cast<idx_t>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<idx_t>(indices.size()),
                     static_cast<idx_t>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[current].escape = static_cast<idx_t>(nodes.size());
}

struct DeviceTriangle {
    Vec3 a, ab, ac, normal, center;
    val_t area;
};

constexpr int MT_WORDS = 624;
constexpr int MT_MIDDLE = 397;
constexpr int MT_INDEPENDENT = MT_WORDS - MT_MIDDLE;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr int PAIRS_PER_TILE = 32;
constexpr int FORM_THREADS = 128;

__host__ __device__ uint32_t mtNext(uint32_t a, uint32_t b, uint32_t middle) {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return middle ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
}

struct FormTile {
    uint32_t mask;
    uint32_t words[MT_WORDS];
};

// The rolling, untempered MT19937 state begins at the next draw. Advancing it
// needs no tempering and updates independent runs of up to 227 words at once.
class MersenneStream {
public:
    MersenneStream() {
        words[0] = 42;
        for (int i = 1; i < MT_WORDS; ++i) {
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        }
        advance(MT_WORDS);
    }

    void advance(uint64_t count) {
        while (count > 0) {
            int end = position < MT_INDEPENDENT ? MT_INDEPENDENT : MT_WORDS - 1;
            int step = static_cast<int>(std::min<uint64_t>(count, end - position));
            if (step > 0) {
                int middle = position + (position < MT_INDEPENDENT ? MT_MIDDLE : -MT_INDEPENDENT);
                for (int i = 0; i < step; ++i) {
                    words[position + i] = mtNext(words[position + i], words[position + i + 1],
                                                  words[middle + i]);
                }
                position += step;
                count -= step;
            } else {
                words[MT_WORDS - 1] = mtNext(words[MT_WORDS - 1], words[0], words[MT_MIDDLE - 1]);
                position = 0;
                --count;
            }
        }
    }

    void snapshot(uint32_t* out) const {
        std::copy(words.begin() + position, words.end(), out);
        std::copy(words.begin(), words.begin() + position, out + MT_WORDS - position);
    }

    void seek(uint64_t offset) {
        // Sequential advance is cheaper for nearby offsets.
        if (offset <= (1u << 22)) {
            advance(offset);
            return;
        }

        // Compute x^offset modulo the characteristic polynomial of the
        // one-word MT19937 transition, then apply that polynomial to the state.
        // These are the non-leading exponents of the standard recurrence
        // polynomial (obtainable by Berlekamp-Massey on 2*19937 output bits).
        constexpr int degree = 19937;
        constexpr int exponents[] = {
            0,1189,1416,1585,1643,1870,2493,2773,3000,3227,3454,
            3681,3908,4135,4362,4753,5661,6337,6569,7129,7477,7525,
            7583,7752,7979,8206,9505,9901,9969,10128,10693,10761,10920,
            11089,11147,11157,11215,11321,11374,11384,11485,11611,11712,11717,
            11838,11881,11944,11997,12277,12335,12393,12504,12509,12620,12673,
            12731,12736,12789,12905,12958,12963,13137,13185,13190,13243,13301,
            13412,13528,13533,13639,13697,13760,13813,13866,14093,14151,14209,
            14320,14325,14436,14547,14552,14605,14721,14774,14779,14953,15001,
            15006,15059,15117,15228,15344,15349,15455,15513,15576,15629,15682,
            15909,15967,16025,16136,16141,16252,16363,16368,16421,16537,16590,
            16595,16817,16822,16875,16933,17044,17160,17271,17329,17445,17498,
            17725,17783,17841,17952,18068,18179,18237,18406,18633,18691,18860,
            19087,19314
        };
        constexpr int limbs = (degree + 31) / 32;
        std::array<uint64_t, limbs> squared{};
        squared[0] = 1;
        for (int bit = 63 - __builtin_clzll(offset); bit >= 0; --bit) {
            int shift = (offset >> bit) & 1;
            for (int i = limbs - 1; i >= 0; --i) {
                uint64_t x = static_cast<uint32_t>(squared[i / 2] >> (32 * (i % 2)));
                x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
                x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
                x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
                x = (x | (x << 2)) & 0x3333333333333333ULL;
                x = (x | (x << 1)) & 0x5555555555555555ULL;
                squared[i] = x << shift;
            }
            for (int i = 2 * degree - 1; i >= degree; --i) {
                uint64_t mask = uint64_t{1} << (i % 64);
                if (!(squared[i / 64] & mask)) continue;
                squared[i / 64] ^= mask;
                for (int exponent : exponents) {
                    int target = i - degree + exponent;
                    squared[target / 64] ^= uint64_t{1} << (target % 64);
                }
            }
        }

        std::array<uint32_t, MT_WORDS> next{};
        for (int i = 0; i < degree; ++i) {
            if (squared[i / 64] & (uint64_t{1} << (i % 64))) {
                int tail = MT_WORDS - position;
                for (int j = 0; j < tail; ++j) next[j] ^= words[position + j];
                for (int j = 0; j < position; ++j) next[tail + j] ^= words[j];
            }
            advance(1);
        }
        words = next;
        position = 0;
    }

private:
    std::array<uint32_t, MT_WORDS> words;
    int position = 0;
};

__device__ val_t uniformFloat(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    return fminf(__uint2float_rn(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 randomPoint(const DeviceTriangle& triangle, const uint32_t* words) {
    val_t u = uniformFloat(words[0]);
    val_t v = uniformFloat(words[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + triangle.ab * u + triangle.ac * v;
}

__device__ bool boxIntersects(const DeviceNode& node, const Vec3& midpoint,
                              const Vec3& d, const Vec3& ad) {
    Vec3 c = midpoint - node.center;
    Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z)
        return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    return fabsf(d.x * c.y - d.y * c.x) <= h.x * ad.y + h.y * ad.x + EPSILON;
}

__device__ bool triangleBlocks(const DeviceTriangle& tri, const Vec3& from,
                               const Vec3& direction, val_t length) {
    Vec3 cross = direction.cross(tri.ac);
    val_t det = tri.ab.dot(cross);
    if (fabsf(det) < EPSILON) return false;
    val_t invDet = ONE / det;
    Vec3 delta = from - tri.a;
    val_t u = delta.dot(cross) * invDet;
    if (u < ZERO || u > ONE) return false;
    Vec3 q = delta.cross(tri.ab);
    val_t v = direction.dot(q) * invDet;
    if (v < ZERO || u + v > ONE) return false;
    val_t distance = tri.ac.dot(q) * invDet;
    return distance > EPSILON && distance < length - EPSILON;
}

struct DeviceScene {
    const DeviceTriangle* triangles;
    const DeviceNode* nodes;
    const idx_t* indices;
};

__device__ bool rayBlocked(const DeviceScene& scene, idx_t i, idx_t j,
                           const Vec3& from, const Vec3& ray, val_t length) {
    Vec3 direction = ray / length;
    Vec3 d = ray * 0.5f;
    Vec3 midpoint = from + d;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    idx_t end = scene.nodes[0].escape;
    idx_t first = scene.nodes[0].triangleCount > 0 ? 0 : 1;
    for (idx_t k = first; k < end;) {
        const DeviceNode& node = scene.nodes[k];
        if (k != 0 && !boxIntersects(node, midpoint, d, ad)) {
            k = node.escape;
            continue;
        }
        for (idx_t p = 0; p < node.triangleCount; ++p) {
            idx_t candidate = scene.indices[node.firstTriangle + p];
            if (candidate != i && candidate != j &&
                triangleBlocks(scene.triangles[candidate], from, direction, length)) {
                return true;
            }
        }
        ++k;
    }
    return false;
}

__device__ val_t rayFactor(const DeviceScene& scene, idx_t i, idx_t j, const uint32_t* draws) {
    const auto& a = scene.triangles[i];
    const auto& b = scene.triangles[j];
    Vec3 from = randomPoint(a, draws);
    Vec3 to = randomPoint(b, draws + 2);
    Vec3 ray = to - from;
    val_t squared = ray.squaredNorm();
    if (squared < EPSILON) return ZERO;
    val_t length = sqrtf(squared);
    if (rayBlocked(scene, i, j, from, ray, length)) return ZERO;

    val_t cosineA = ray.dot(a.normal) / length;
    val_t cosineB = (-ray).dot(b.normal) / length;
    if (cosineA <= ZERO || cosineB <= ZERO) return ZERO;
    return (cosineA * cosineB) / (PI * squared);
}

__global__ void formFactors(DeviceScene scene, int n, int firstRow, int firstColumn,
                            int pairs, const FormTile* tiles, val_t* weights, uint8_t* delays,
                            unsigned long long* nonzero, int* minDelay) {
    static_assert(PAIRS_PER_TILE * DRAWS_PER_PAIR >= MT_WORDS);
    __shared__ uint32_t randomWords[PAIRS_PER_TILE * DRAWS_PER_PAIR];
    __shared__ val_t rayFactors[FORM_THREADS];
    __shared__ int tileDelays[PAIRS_PER_TILE];
    __shared__ int tileCounts[PAIRS_PER_TILE];

    int thread = threadIdx.x;
    uint32_t mask = tiles[blockIdx.x].mask;
    int draws = __popc(mask) * DRAWS_PER_PAIR;
    if (draws > 0) {
        for (int i = thread; i < MT_WORDS; i += FORM_THREADS) {
            randomWords[i] = tiles[blockIdx.x].words[i];
        }
        __syncthreads();
        // Each new word depends on words at least 227 positions earlier.
        for (int first = MT_WORDS; first < draws; first += MT_INDEPENDENT) {
            int end = min(first + MT_INDEPENDENT, draws);
            for (int i = first + thread; i < end; i += FORM_THREADS) {
                randomWords[i] = mtNext(randomWords[i - MT_WORDS], randomWords[i - MT_WORDS + 1],
                                        randomWords[i - MT_INDEPENDENT]);
            }
            __syncthreads();
        }
    }

    int tileBase = blockIdx.x * PAIRS_PER_TILE;
    int ray = thread % NUM_RAYS;
    for (int p = thread / NUM_RAYS; p < PAIRS_PER_TILE; p += FORM_THREADS / NUM_RAYS) {
        int pair = tileBase + p;
        int position = pair + firstColumn;
        uint32_t bit = uint32_t{1} << p;
        bool active = mask & bit;
        idx_t i = firstRow + position / n;
        idx_t j = position % n;
        val_t factor = ZERO;
        if (active) {
            int sample = __popc(mask & (bit - 1)) * DRAWS_PER_PAIR + ray * 4;
            factor = rayFactor(scene, i, j, randomWords + sample);
        }
        rayFactors[thread] = factor;
        __syncwarp();
        if (ray == 0) {
            val_t kij = ZERO;
            #pragma unroll
            for (int r = 0; r < NUM_RAYS; ++r) kij += rayFactors[thread + r];
            kij *= INV_NUM_RAYS;
            int delay = 0;
            val_t weight = ZERO;
            if (kij > ZERO) {
                const auto& a = scene.triangles[i];
                const auto& b = scene.triangles[j];
                // Distinct centers on the generated mesh give positive delays.
                delay = static_cast<int>(ceilf((a.center - b.center).norm() * INV_WAVE_SPEED));
                weight = fminf(kij * b.area, ONE);
            }
            if (pair < pairs) {
                weights[pair] = weight;
                delays[pair] = static_cast<uint8_t>(delay);
            }
            tileDelays[p] = delay > 0 ? delay : INT_MAX;
            tileCounts[p] = kij > EPSILON;
        }
        __syncwarp();
    }
    if (!minDelay && !nonzero) return;
    __syncthreads();
    if (thread < PAIRS_PER_TILE) {
        int best = tileDelays[thread];
        int count = tileCounts[thread];
        for (int d = 16; d; d /= 2) {
            best = min(best, __shfl_down_sync(0xffffffffu, best, d));
            count += __shfl_down_sync(0xffffffffu, count, d);
        }
        if (thread == 0) {
            if (minDelay && best != INT_MAX) atomicMin(minDelay, best);
            if (nonzero && count) atomicAdd(nonzero, static_cast<unsigned long long>(count));
        }
    }
}

// ============================================================================
constexpr int PROPAGATION_THREADS = 128;

template<bool Distributed>
__global__ void propagate(int n, int firstRow, int outputStride, int ringSteps, int time,
                          int sourceIndex, int timeOff, val_t rho, const val_t* weights,
                          const uint8_t* delays, const val_t* preceding, val_t* output) {
    __shared__ val_t contributions[PROPAGATION_THREADS];
    int thread = threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.x) * n;
    val_t sum = ZERO;

    for (int offset = 0; offset < n; offset += PROPAGATION_THREADS) {
        int j = offset + thread;
        val_t contribution = ZERO;
        if (j < n) {
            int delay = delays[row + j];
            if (delay > 0 && delay <= time) {
                int past = time - delay;
                if constexpr (Distributed) past %= ringSteps;
                val_t rad = preceding[static_cast<size_t>(past) * n + j];
                if (!(rad <= ZERO)) contribution = weights[row + j] * rad;
            }
        }
        contributions[thread] = contribution;
        if (__syncthreads_or(contribution != ZERO)) {
            // Keep the original ascending-j sum order.
            if (thread == 0) {
                #pragma unroll 8
                for (int k = 0; k < PROPAGATION_THREADS; ++k) sum += contributions[k];
            }
            __syncthreads();
        }
    }

    if (thread == 0) {
        int i = firstRow + blockIdx.x;
        val_t emission = i == sourceIndex && time < timeOff ? ONE : ZERO;
        output[static_cast<size_t>(time) * outputStride + blockIdx.x] = fmaf(rho, sum, emission);
    }
}

// The all-gather packs each rank's rectangular time-by-row slab contiguously.
// A single pass restores the time-major layout used by the propagation kernel.
__global__ void unpackHistory(int n, int ranks, int start, int count, int ringSteps,
                               int sourceIndex, const val_t* packed, val_t* history,
                               val_t* sourceHistory) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n * count) return;
    int t = pos / n;
    int j = pos % n;
    int base = n / ranks;
    int remainder = n % ranks;
    int wide = (base + 1) * remainder;
    int width, first;
    if (j < wide) {
        width = base + 1;
        first = j / width * width;
    } else {
        width = base;
        first = wide + (j - wide) / width * width;
    }
    val_t value = packed[first * count + t * width + (j - first)];
    int time = start + t;
    history[static_cast<size_t>(time % ringSteps) * n + j] = value;
    if (j == sourceIndex) sourceHistory[time] = value;
}

constexpr int CORRELATION_THREADS = 128;
constexpr int CACHED_TIMESTEPS = 1024;

__global__ void correlate(int width, int timesteps, const val_t* history,
                          const val_t* source, int sourceStride, val_t* distances, int* received) {
    __shared__ val_t maxima[CORRELATION_THREADS];
    __shared__ int lags[CORRELATION_THREADS];
    extern __shared__ val_t signals[];
    int thread = threadIdx.x;
    size_t row = blockIdx.x;
    if (timesteps > 0 && timesteps <= CACHED_TIMESTEPS) {
        for (int t = thread; t < timesteps; t += CORRELATION_THREADS) {
            signals[t] = source[static_cast<size_t>(t) * sourceStride];
            signals[timesteps + t] = history[static_cast<size_t>(t) * width + row];
        }
        __syncthreads();
        source = signals;
        sourceStride = 1;
        history = signals + timesteps;
        width = 1;
        row = 0;
    }
    val_t maximum = ZERO;
    int bestLag = 0;
    bool hasEnergy = false;
    for (int lag = thread; lag < timesteps; lag += CORRELATION_THREADS) {
        val_t sum = ZERO;
        for (int time = lag; time < timesteps; ++time) {
            val_t a = source[static_cast<size_t>(time - lag) * sourceStride];
            val_t b = history[static_cast<size_t>(time) * width + row];
            sum += a * b;
        }
        if (sum > maximum) {
            maximum = sum;
            bestLag = lag;
        }
        if (received && history[static_cast<size_t>(lag) * width + row] > EPSILON) hasEnergy = true;
    }
    maxima[thread] = maximum;
    lags[thread] = bestLag;
    int energy = __syncthreads_or(hasEnergy);
    for (int step = CORRELATION_THREADS / 2; step > 0; step /= 2) {
        if (thread < step) {
            val_t other = maxima[thread + step];
            int lag = lags[thread + step];
            if (other > maximum || (other == maximum && lag < bestLag)) {
                maxima[thread] = maximum = other;
                lags[thread] = bestLag = lag;
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        distances[blockIdx.x] = WAVE_SPEED * static_cast<val_t>(bestLag);
        if (energy) atomicAdd(received, 1);
    }
}

struct DistributedSimulation {
    DistributedSimulation(int triangles, int timesteps, size_t source, val_t reflectivity,
                          const MpiSession& mpi)
        : n(triangles), steps(timesteps), sourceIndex(static_cast<int>(source % n)),
          rho(reflectivity), rows(n, mpi), pairs(static_cast<size_t>(rows.count) * n),
          weights(pairs), delays(pairs), history(static_cast<size_t>(steps) * rows.count),
          ringSteps(mpi.size > 1 ? std::min(steps, MAX_DELAY) : 0),
          ring(static_cast<size_t>(ringSteps) * n), sourceHistory(mpi.size > 1 ? steps : 0) {}

    int n, steps, sourceIndex;
    val_t rho;
    RowPartition rows;
    size_t pairs;
    CudaBuffer<val_t> weights;
    CudaBuffer<uint8_t> delays;
    CudaBuffer<val_t> history;
    int ringSteps;
    CudaBuffer<val_t> ring;
    CudaBuffer<val_t> sourceHistory;
    unsigned long long localNonzero = 0;
    int localReceived = 0;
    int window = 1;

    void precompute(const std::vector<Triangle>& triangles, const Octree& octree,
                    const MpiSession& mpi, bool validate) {
        std::vector<DeviceTriangle> geometry(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const auto& t = triangles[i];
            geometry[i] = {t.a, t.b - t.a, t.c - t.a, t.normal(), t.center(), t.area()};
        }

        size_t numTiles = (pairs + PAIRS_PER_TILE - 1) / PAIRS_PER_TILE;
        std::vector<uint32_t> masks(numTiles);
        unsigned long long localDraws = 0;
        #pragma omp parallel for schedule(static) reduction(+:localDraws) if(numTiles >= 512)
        for (size_t tile = 0; tile < numTiles; ++tile) {
            size_t pair = tile * PAIRS_PER_TILE;
            int i = rows.first + static_cast<int>(pair / n);
            int j = static_cast<int>(pair % n);
            uint32_t mask = 0;
            for (int bit = 0; bit < PAIRS_PER_TILE && pair + bit < pairs; ++bit) {
                if (i != j && geometry[i].normal.dot(geometry[j].normal) <= 0.99f)
                    mask |= uint32_t{1} << bit;
                if (++j == n) {
                    j = 0;
                    ++i;
                }
            }
            masks[tile] = mask;
            localDraws += __builtin_popcount(mask) * DRAWS_PER_PAIR;
        }

        unsigned long long rankOffset = 0;
        // Only accepted pairs consume draws, in global row-major order.
        MPI_Exscan(&localDraws, &rankOffset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (mpi.rank == 0) rankOffset = 0;

        int smallest = INT_MAX;
        if (pairs > 0) {
            runFormFactorBatches(geometry, octree, masks, rankOffset, smallest, mpi.size > 1, validate);
        }
        if (mpi.size > 1) {
            int globalMin;
            MPI_Allreduce(&smallest, &globalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            window = std::max(1, std::min(globalMin, ringSteps));
        }
    }

    void simulate(const MpiSession& mpi) {
        if (steps == 0) return;
        bool distributed = mpi.size > 1;
        size_t slots = distributed ? static_cast<size_t>(window) * n : 0;
        CudaBuffer<val_t, true> exchange(slots);
        CudaBuffer<val_t> packed(slots);
        std::vector<int> counts(mpi.size), offsets(mpi.size);
        for (int start = 0; start < steps;) {
            int count = distributed ? std::min(window, steps - start) : steps;
            for (int t = start; t < start + count; ++t) {
                if (rows.count > 0) {
                    if (distributed) {
                        propagate<true><<<rows.count, PROPAGATION_THREADS>>>(
                            n, rows.first, rows.count, ringSteps, t, sourceIndex, steps / 2,
                            rho, weights.data, delays.data, ring.data, history.data);
                    } else {
                        propagate<false><<<rows.count, PROPAGATION_THREADS>>>(
                            n, 0, n, 0, t, sourceIndex, steps / 2, rho,
                            weights.data, delays.data, history.data, history.data);
                    }
                }
                if (mpi.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == steps))
                    printf("  Timestep %d/%d\n", t + 1, steps);
            }
            CUDA_CHECK(cudaGetLastError());
            if (distributed) {
                for (int r = 0; r < mpi.size; ++r) {
                    counts[r] = count * rows.counts[r];
                    offsets[r] = count * rows.offsets[r];
                }
                if (rows.count > 0) {
                    CUDA_CHECK(cudaMemcpy(exchange.data + offsets[mpi.rank],
                        history.data + static_cast<size_t>(start) * rows.count,
                        counts[mpi.rank] * sizeof(val_t), cudaMemcpyDeviceToHost));
                }
                // All dependencies reach back beyond this window, so one
                // collective suffices for its entire set of timesteps.
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange.data,
                               counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
                int elements = count * n;
                CUDA_CHECK(cudaMemcpy(packed.data, exchange.data, elements * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
                constexpr int threads = 256;
                unpackHistory<<<(elements + threads - 1) / threads, threads>>>(
                    n, mpi.size, start, count, ringSteps, sourceIndex,
                    packed.data, ring.data, sourceHistory.data);
                CUDA_CHECK(cudaGetLastError());
            }
            start += count;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<val_t> distances(const MpiSession& mpi, bool validate) {
        std::vector<val_t> local(rows.count);
        if (rows.count > 0) {
            CudaBuffer<val_t> deviceDistances(rows.count);
            CudaBuffer<int> received(validate ? 1 : 0);
            if (validate) CUDA_CHECK(cudaMemset(received.data, 0, sizeof(int)));
            bool distributed = mpi.size > 1;
            const val_t* source = distributed ? sourceHistory.data : history.data;
            if (!distributed && steps > 0) source += sourceIndex;
            size_t shared = steps <= CACHED_TIMESTEPS ? 2 * steps * sizeof(val_t) : 0;
            correlate<<<rows.count, CORRELATION_THREADS, shared>>>(
                rows.count, steps, history.data, source, distributed ? 1 : n,
                deviceDistances.data, received.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local.data(), deviceDistances.data, local.size() * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
            if (validate) CUDA_CHECK(cudaMemcpy(&localReceived, received.data, sizeof(int), cudaMemcpyDeviceToHost));
        }

        std::vector<val_t> result(mpi.rank == 0 ? n : 0);
        MPI_Gatherv(local.data(), rows.count, MPI_FLOAT, result.data(), rows.counts.data(),
                    rows.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return result;
    }

private:
    void runFormFactorBatches(const std::vector<DeviceTriangle>& geometry, const Octree& octree,
                              const std::vector<uint32_t>& masks, uint64_t rankOffset,
                              int& smallest, bool distributed, bool validate) {
        std::vector<DeviceNode> nodes;
        std::vector<idx_t> indices;
        flattenOctree(octree, nodes, indices);
        CudaBuffer<DeviceTriangle> deviceTriangles(geometry.size());
        CudaBuffer<DeviceNode> deviceNodes(nodes.size());
        CudaBuffer<idx_t> deviceIndices(indices.size());
        CUDA_CHECK(cudaMemcpy(deviceTriangles.data, geometry.data(), geometry.size() * sizeof(DeviceTriangle),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceNodes.data, nodes.data(), nodes.size() * sizeof(DeviceNode),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceIndices.data, indices.data(), indices.size() * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
        DeviceScene scene{deviceTriangles.data, deviceNodes.data, deviceIndices.data};

        CudaBuffer<unsigned long long> nonzero(validate ? 1 : 0);
        CudaBuffer<int> minDelay(distributed ? 1 : 0);
        if (validate) CUDA_CHECK(cudaMemset(nonzero.data, 0, sizeof(localNonzero)));
        if (distributed) CUDA_CHECK(cudaMemcpy(minDelay.data, &smallest, sizeof(int), cudaMemcpyHostToDevice));

        // Stage checkpoints in bounded batches. Two slots let the host
        // prepare checkpoints and the GPU copy and trace independently.
        size_t capacity = std::min<size_t>(2048, masks.size());
        int slots = masks.size() > capacity ? 2 : 1;
        CudaBuffer<FormTile, true> hostTiles(slots * capacity);
        CudaBuffer<FormTile> deviceTiles(slots * capacity);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        cudaStream_t streams[2];
        cudaEvent_t ready[2];
        for (int slot = 0; slot < slots; ++slot) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[slot], cudaEventDisableTiming));
        }

        MersenneStream rng;
        rng.seek(rankOffset);
        for (size_t first = 0, batch = 0; first < masks.size(); first += capacity, ++batch) {
            int slot = batch % slots;
            auto stream = streams[slot];
            FormTile* host = hostTiles.data + slot * capacity;
            FormTile* device = deviceTiles.data + slot * capacity;
            size_t count = std::min(capacity, masks.size() - first);
            if (batch >= static_cast<size_t>(slots)) CUDA_CHECK(cudaEventSynchronize(ready[slot]));
            for (size_t tile = 0; tile < count; ++tile) {
                host[tile].mask = masks[first + tile];
                if (host[tile].mask) {
                    rng.snapshot(host[tile].words);
                    rng.advance(__builtin_popcount(host[tile].mask) * DRAWS_PER_PAIR);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(device, host, count * sizeof(FormTile), cudaMemcpyHostToDevice, stream));
            size_t offset = first * PAIRS_PER_TILE;
            int firstRow = rows.first + static_cast<int>(offset / n);
            int firstColumn = static_cast<int>(offset % n);
            int batchPairs = static_cast<int>(std::min(pairs - offset, count * PAIRS_PER_TILE));
            formFactors<<<static_cast<unsigned>(count), FORM_THREADS, 0, stream>>>(
                scene, n, firstRow, firstColumn, batchPairs, device,
                weights.data + offset, delays.data + offset, nonzero.data, minDelay.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ready[slot], stream));
        }
        for (int slot = 0; slot < slots; ++slot) {
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            CUDA_CHECK(cudaEventDestroy(ready[slot]));
            CUDA_CHECK(cudaStreamDestroy(streams[slot]));
        }

        if (validate) CUDA_CHECK(cudaMemcpy(&localNonzero, nonzero.data, sizeof(localNonzero), cudaMemcpyDeviceToHost));
        if (distributed) CUDA_CHECK(cudaMemcpy(&smallest, minDelay.data, sizeof(int), cudaMemcpyDeviceToHost));
    }
};

bool validateResults(const std::vector<val_t>& distances, int sourceIndex, int receivedEnergy,
                     unsigned long long nonzeroKij) {
    printf("\nValidation:\n");
    size_t n = distances.size();
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    for (size_t i = 0; i < n; ++i) {
        val_t d = distances[i];
        if (d < ZERO) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu\n", i);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) ++nonZeroCount;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);
    if (distances[sourceIndex] > WAVE_SPEED * 2)
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", distances[sourceIndex]);

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, n);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonzeroKij, n * n, 100.0f * nonzeroKij / static_cast<val_t>(n * n));
    if (nonzeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (allNonNegative) printf("  Validation: PASSED\n");
    return allNonNegative;
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

long elapsedMilliseconds(double start) {
    double elapsed = MPI_Wtime() - start;
    double maximum = 0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return static_cast<long>(1000.0 * maximum);
}

} // namespace

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

int runBenchmark(int argc, char** argv, const MpiSession& mpi) {
    bool root = mpi.rank == 0;
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (root) fprintf(stderr, "Number of timesteps must be non-negative\n");
        return EXIT_FAILURE;
    }

    mpi.selectDevice();
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_max_threads()));
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

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
        printf("Generated icosphere mesh with %zu triangles\n", mesh.triangles.size());
        printf("Building octree...\n\n");
    }
    Octree octree;
    octree.build(mesh.triangles);
    DistributedSimulation state(static_cast<int>(mesh.triangles.size()), timesteps,
                                static_cast<size_t>(sourceIdx), reflectivity, mpi);

    if (root) printf("Computing time delays (Tau) and form factors (Kij)...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    state.precompute(mesh.triangles, octree, mpi, validate);
    long preDuration = elapsedMilliseconds(start);
    if (root) {
        printf("Precomputation time: %ld ms\n\n", preDuration);
        printf("Running wave propagation simulation...\n");
    }

    start = MPI_Wtime();
    state.simulate(mpi);
    long simDuration = elapsedMilliseconds(start);
    if (root) {
        printf("Simulation time: %ld ms\n\n", simDuration);
        printf("Computing distances via cross-correlation...\n");
    }

    start = MPI_Wtime();
    std::vector<val_t> distances = state.distances(mpi, validate);
    long distDuration = elapsedMilliseconds(start);

    if (root) {
        long totalTime = preDuration + simDuration + distDuration;
        printf("Distance computation time: %ld ms\n\n", distDuration);
        printf("Total computation time: %ld ms\n", totalTime);

        size_t n = state.n;
        size_t t = state.steps;
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        size_t memory = n * n * (sizeof(val_t) + sizeof(uint8_t)) + t * n * sizeof(val_t);
        if (mpi.size > 1) memory += mpi.size * (state.ringSteps * n + t) * sizeof(val_t);
        printf("  Memory usage: %.2f MB\n", memory / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", computeHash(distances));
        if (printResults) {
            std::vector<double> distData(distances.begin(), distances.end());
            print_results(distData, "Distances");
        }
    }

    int result = EXIT_SUCCESS;
    if (validate) {
        int received = 0;
        unsigned long long nonzero = 0;
        MPI_Reduce(&state.localReceived, &received, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&state.localNonzero, &nonzero, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (root && !validateResults(distances, state.sourceIndex, received, nonzero))
            result = EXIT_FAILURE;
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return result;
}

int main(int argc, char** argv) {
    MpiSession mpi(argc, argv);
    try {
        return runBenchmark(argc, argv, mpi);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", mpi.rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}
