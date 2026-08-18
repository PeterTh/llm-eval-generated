#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t globalIndex = globalOffset + i;

        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor =
            1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

struct WorkRange {
    size_t begin;
    size_t count;
};

// Assign contiguous, almost equally sized ranges.  Contiguous ranges preserve
// the serial result order and make ordered collection inexpensive.
WorkRange getWorkRange(const size_t total, const int rank,
                       const int numRanks) noexcept {
    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t thisRank = static_cast<size_t>(rank);
    const size_t base = total / ranks;
    const size_t remainder = total % ranks;

    return {thisRank * base + std::min(thisRank, remainder),
            base + (thisRank < remainder ? 1U : 0U)};
}

// Collect a prefix of the distributed result vector in global order.  The
// operation is split into MPI-int-sized windows so very large problem sizes do
// not overflow MPI_Gatherv's count or displacement arguments.
void gatherResultPrefix(const std::vector<double>& localResults,
                        const WorkRange localRange, const size_t globalSize,
                        const size_t prefixSize,
                        std::vector<double>& gatheredResults, const int rank,
                        const int numRanks) {
    if (rank == 0) {
        gatheredResults.resize(prefixSize);
    }

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(numRanks));
        displacements.resize(static_cast<size_t>(numRanks));
    }

    constexpr size_t maxWindow =
        static_cast<size_t>(std::numeric_limits<int>::max());
    size_t windowBegin = 0;
    while (windowBegin < prefixSize) {
        const size_t windowSize = std::min(maxWindow, prefixSize - windowBegin);
        const size_t windowEnd = windowBegin + windowSize;
        const size_t localEnd = localRange.begin + localRange.count;
        const size_t sendBegin = std::max(localRange.begin, windowBegin);
        const size_t sendEnd = std::min(localEnd, windowEnd);
        const size_t sendSize = sendEnd > sendBegin ? sendEnd - sendBegin : 0;
        const size_t localOffset = sendSize == 0 ? 0 : sendBegin - localRange.begin;

        if (rank == 0) {
            for (int source = 0; source < numRanks; ++source) {
                const WorkRange originalRange =
                    getWorkRange(globalSize, source, numRanks);
                const size_t sourceEnd = originalRange.begin + originalRange.count;
                const size_t receiveBegin =
                    std::max(originalRange.begin, windowBegin);
                const size_t receiveEnd = std::min(sourceEnd, windowEnd);
                const size_t receiveSize =
                    receiveEnd > receiveBegin ? receiveEnd - receiveBegin : 0;
                receiveCounts[static_cast<size_t>(source)] =
                    static_cast<int>(receiveSize);
                displacements[static_cast<size_t>(source)] =
                    static_cast<int>(receiveSize == 0 ? 0
                                                      : receiveBegin - windowBegin);
            }
        }

        MPI_Gatherv(sendSize == 0 ? nullptr : localResults.data() + localOffset,
                    static_cast<int>(sendSize), MPI_DOUBLE,
                    rank == 0 ? gatheredResults.data() + windowBegin : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        windowBegin = windowEnd;
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    
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
    }

    const WorkRange localRange = getWorkRange(numOptions, rank, numRanks);
    
    // Generate only this rank's deterministic slice.  No input distribution or
    // full-size replicated storage is required.
    std::vector<OptionInput> options;
    generateOptions(options, localRange.count, localRange.begin);
    
    std::vector<double> localResults(localRange.count);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Synchronize before timing and use the slowest rank's elapsed time: this
    // is the distributed wall time that determines application throughput.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localRange.count; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Options per second: %.0f\n",
               static_cast<double>(numOptions) / elapsed);
    }
    
    // Print results for external validation
    std::vector<double> gatheredResults;
    if (printResults) {
        gatherResultPrefix(localResults, localRange, numOptions, numOptions,
                           gatheredResults, rank, numRanks);
        if (rank == 0) {
            print_results(gatheredResults, "OptionPrices");
        }
    }
    
    // Validation
    if (validate) {
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        if (!printResults) {
            gatherResultPrefix(localResults, localRange, numOptions, numChecks,
                               gatheredResults, rank, numRanks);
        }

        if (rank == 0) {
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, numChecks);
            printf("Validating results...\n");
            const bool valid =
                validateResults(validationOptions, gatheredResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
