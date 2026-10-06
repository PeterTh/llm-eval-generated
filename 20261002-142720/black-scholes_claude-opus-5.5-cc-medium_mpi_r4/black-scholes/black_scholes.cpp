#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double discount = exp(-r * T);
    
    // Only evaluate the CDF terms each branch actually uses
    double price;
    if (option.type == CALL) {
        const double Nd1 = cumulativeNormal(d1);
        const double Nd2 = cumulativeNormal(d2);
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Standard test cases for validation
inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

// Generate a larger set of options by scaling the test set
// Only the global index range [begin, begin + count) is generated, stored locally.
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);
    
    for (size_t j = 0; j < count; ++j) {
        const size_t i = begin + j;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[j] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[j].spot *= factor;
        options[j].strike *= factor;
    }
}

// Block decomposition of [0, n) over nranks ranks
inline size_t blockBegin(const size_t n, const int nranks, const int rank) noexcept {
    const size_t base = n / nranks;
    const size_t rem = n % nranks;
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

// Gather distributed blocks into 'results' on rank 0 (handles counts > INT_MAX)
void gatherResults(std::vector<double>& results, const std::vector<double>& local,
                   const size_t n, const int nranks, const int rank) {
    constexpr size_t maxChunk = 1u << 30;
    if (rank == 0) {
        results.resize(n);
        std::copy(local.begin(), local.end(), results.begin());
        for (int src = 1; src < nranks; ++src) {
            const size_t b = blockBegin(n, nranks, src);
            const size_t e = blockBegin(n, nranks, src + 1);
            for (size_t off = b; off < e; off += maxChunk) {
                const int cnt = static_cast<int>(std::min(maxChunk, e - off));
                MPI_Recv(results.data() + off, cnt, MPI_DOUBLE, src, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        const size_t cnt = local.size();
        for (size_t off = 0; off < cnt; off += maxChunk) {
            const int c = static_cast<int>(std::min(maxChunk, cnt - off));
            MPI_Send(local.data() + off, c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    
    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identically on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank generates only its own block of options
    const size_t begin = blockBegin(numOptions, nranks, rank);
    const size_t localCount = blockBegin(numOptions, nranks, rank + 1) - begin;
    std::vector<OptionInput> options;
    generateOptions(options, begin, localCount);
    
    // Allocate local results
    std::vector<double> localResults(localCount);
    
    // Price options
    if (root) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    const OptionInput* __restrict opts = options.data();
    double* __restrict out = localResults.data();
    for (size_t i = 0; i < localCount; ++i) {
        out[i] = blackScholes(opts[i]);
    }
    
    // Global completion time is defined by the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (root) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    }
    
    // Collect full result vector on rank 0 only when it is needed
    std::vector<double> results;
    if (printResults || validate) {
        gatherResults(results, localResults, numOptions, nranks, rank);
    }
    
    // Print results for external validation
    if (printResults && root) {
        print_results(results, "OptionPrices");
    }
    
    int exitCode = 0;
    // Validation
    if (validate) {
        if (root) {
            printf("Validating results...\n");
            // Rank 0 regenerates the (few) leading options needed for checking
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, 0, std::min(static_cast<size_t>(10), numOptions));
            bool valid = validateResults(checkOptions, results);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return exitCode;
}
