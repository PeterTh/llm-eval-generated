#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

constexpr unsigned int INF = 1000000000;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    for (size_t k = 0; k < numNodes; ++k) {
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

int main() {
    size_t numNodes = 512;
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize with simple values
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = i % 100 + 1;
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        path[i] = i % numNodes;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("CPU Floyd-Warshall time: %ld ms\n", duration.count());
    
    return 0;
}
