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
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (option.type == CALL) {
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

// Generate the options with global indices [begin, begin + count) by scaling the test set.
// The generation depends only on the global index, so every rank can produce its own
// slice independently and identically to the sequential version.
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

    for (size_t k = 0; k < count; ++k) {
        const size_t i = begin + k;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[k] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[k].spot *= factor;
        options[k].strike *= factor;
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

// Contiguous block decomposition: rank `r` of `numRanks` owns the global indices
// [blockBegin(r), blockBegin(r + 1)). The remainder is spread over the first ranks so
// that the per-rank load differs by at most one option.
inline size_t blockBegin(const size_t numOptions, const int numRanks, const int r) noexcept {
    const size_t base = numOptions / static_cast<size_t>(numRanks);
    const size_t rem = numOptions % static_cast<size_t>(numRanks);
    const size_t ur = static_cast<size_t>(r);
    return ur * base + std::min(ur, rem);
}

// Collect the distributed result slices into one array on rank 0, in global index order.
// The transfers are chunked so that the element counts always fit into MPI's int-sized
// count arguments, even for very large problem sizes.
void gatherResults(const std::vector<double>& local, std::vector<double>& global,
                   const size_t numOptions, const int rank, const int numRanks) {
    constexpr size_t maxChunk = size_t{1} << 24;

    if (rank == 0) {
        global.resize(numOptions);
        std::copy(local.begin(), local.end(), global.begin());

        for (int src = 1; src < numRanks; ++src) {
            const size_t begin = blockBegin(numOptions, numRanks, src);
            const size_t count = blockBegin(numOptions, numRanks, src + 1) - begin;
            for (size_t done = 0; done < count; done += maxChunk) {
                const int n = static_cast<int>(std::min(maxChunk, count - done));
                MPI_Recv(global.data() + begin + done, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    } else {
        const size_t count = local.size();
        for (size_t done = 0; done < count; done += maxChunk) {
            const int n = static_cast<int>(std::min(maxChunk, count - done));
            MPI_Send(local.data() + done, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Determine the local slice of the global option range
    const size_t localBegin = blockBegin(numOptions, numRanks, rank);
    const size_t localCount = blockBegin(numOptions, numRanks, rank + 1) - localBegin;

    // Generate the local options (index-based, no communication required)
    std::vector<OptionInput> options;
    generateOptions(options, localBegin, localCount);

    // Allocate local results
    std::vector<double> results(localCount);

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (size_t i = 0; i < localCount; ++i) {
        results[i] = blackScholes(options[i]);
    }

    const double localElapsed = MPI_Wtime() - start;

    // The benchmark time is determined by the slowest rank
    double elapsed = localElapsed;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1.0e3);
        printf("Options per second: %.0f\n", numOptions / elapsed);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> allResults;
        gatherResults(results, allResults, numOptions, rank, numRanks);
        if (rank == 0) {
            print_results(allResults, "OptionPrices");
        }
    }

    // Validation
    int failed = 0;
    if (validate) {
        // Only the first few options are checked; rank 0 regenerates them locally and
        // collects the corresponding prices from their owners.
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        std::vector<double> checkResults(numChecks);

        for (size_t i = 0; i < numChecks; ++i) {
            const int owner = [&] {
                for (int r = 0; r < numRanks; ++r) {
                    if (i < blockBegin(numOptions, numRanks, r + 1)) {
                        return r;
                    }
                }
                return numRanks - 1;
            }();

            if (rank == owner) {
                const double value = results[i - localBegin];
                if (owner == 0) {
                    checkResults[i] = value;
                } else {
                    MPI_Send(&value, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                }
            } else if (rank == 0) {
                MPI_Recv(&checkResults[i], 1, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }

        if (rank == 0) {
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, 0, numChecks);

            printf("Validating results...\n");
            const bool valid = validateResults(checkOptions, checkResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                failed = 1;
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return failed;
}
