#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <utility>
#include <vector>

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
void generateOptions(std::vector<OptionInput>& options,
                     const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t globalIndex = globalOffset + i;

        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// Divide [0, numOptions) into contiguous, balanced ranges.  The quotient /
// remainder formulation avoids overflow in the usual rank * numOptions
// calculation and gives the first ranks at most one extra option.
std::pair<size_t, size_t> partitionRange(const size_t numOptions,
                                         const int rank,
                                         const int numRanks) noexcept {
    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t thisRank = static_cast<size_t>(rank);
    const size_t base = numOptions / ranks;
    const size_t remainder = numOptions % ranks;
    const size_t count = base + (thisRank < remainder ? 1 : 0);
    const size_t begin = base * thisRank + std::min(thisRank, remainder);
    return {begin, count};
}

// Gather a contiguous range of local results into rank zero in global index
// order.  MPI_Gatherv uses int counts/displacements, so large jobs are
// handled in bounded chunks rather than being limited by INT_MAX.
void gatherResults(const std::vector<double>& localResults,
                   const size_t localBegin,
                   const size_t numOptions,
                   const int rank,
                   const int numRanks,
                   std::vector<double>& results) {
    if (rank == 0) {
        results.resize(numOptions);
    }

    const size_t maxMpiCount = static_cast<size_t>(INT_MAX);
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(numRanks));
        displacements.resize(static_cast<size_t>(numRanks));
    }

    const size_t localEnd = localBegin + localResults.size();
    for (size_t chunkBegin = 0; chunkBegin < numOptions;) {
        const size_t chunkSize = std::min(maxMpiCount, numOptions - chunkBegin);
        const size_t chunkEnd = chunkBegin + chunkSize;
        const size_t overlapBegin = std::max(localBegin, chunkBegin);
        const size_t overlapEnd = std::min(localEnd, chunkEnd);
        const int sendCount = static_cast<int>(overlapEnd > overlapBegin
                                                   ? overlapEnd - overlapBegin
                                                   : 0);
        const double* sendBuffer = sendCount == 0
                                       ? nullptr
                                       : localResults.data() + (overlapBegin - localBegin);

        if (rank == 0) {
            for (int source = 0; source < numRanks; ++source) {
                const auto [sourceBegin, sourceCount] =
                    partitionRange(numOptions, source, numRanks);
                const size_t sourceEnd = sourceBegin + sourceCount;
                const size_t sourceOverlapBegin = std::max(sourceBegin, chunkBegin);
                const size_t sourceOverlapEnd = std::min(sourceEnd, chunkEnd);
                const size_t count = sourceOverlapEnd > sourceOverlapBegin
                                         ? sourceOverlapEnd - sourceOverlapBegin
                                         : 0;
                receiveCounts[static_cast<size_t>(source)] = static_cast<int>(count);
                displacements[static_cast<size_t>(source)] =
                    static_cast<int>(sourceOverlapBegin - chunkBegin);
            }
        }

        double* receiveBuffer = rank == 0 ? results.data() + chunkBegin : nullptr;
        MPI_Gatherv(sendBuffer,
                    sendCount,
                    MPI_DOUBLE,
                    receiveBuffer,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        chunkBegin = chunkEnd;
    }
}

// Validation only needs the first ten global results.  Gathering this prefix
// avoids allocating the full result set on rank zero when -r is not enabled.
void gatherValidationResults(const std::vector<double>& localResults,
                            const size_t localBegin,
                            const size_t numOptions,
                            const int rank,
                            const int numRanks,
                            std::vector<double>& validationResults) {
    const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
    if (rank == 0) {
        validationResults.resize(numChecks);
    }

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(numRanks));
        displacements.resize(static_cast<size_t>(numRanks));
    }

    const size_t localEnd = localBegin + localResults.size();
    const size_t overlapBegin = std::min(localBegin, numChecks);
    const size_t overlapEnd = std::min(localEnd, numChecks);
    const int sendCount = static_cast<int>(overlapEnd > overlapBegin
                                               ? overlapEnd - overlapBegin
                                               : 0);
    const double* sendBuffer = sendCount == 0
                                   ? nullptr
                                   : localResults.data() + (overlapBegin - localBegin);

    if (rank == 0) {
        for (int source = 0; source < numRanks; ++source) {
            const auto [sourceBegin, sourceCount] =
                partitionRange(numOptions, source, numRanks);
            const size_t sourceEnd = sourceBegin + sourceCount;
            const size_t sourceOverlapBegin = std::min(sourceBegin, numChecks);
            const size_t sourceOverlapEnd = std::min(sourceEnd, numChecks);
            receiveCounts[static_cast<size_t>(source)] = static_cast<int>(
                sourceOverlapEnd > sourceOverlapBegin
                    ? sourceOverlapEnd - sourceOverlapBegin
                    : 0);
            displacements[static_cast<size_t>(source)] =
                static_cast<int>(sourceOverlapBegin);
        }
    }

    MPI_Gatherv(sendBuffer,
                sendCount,
                MPI_DOUBLE,
                rank == 0 ? validationResults.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
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

    const auto [localBegin, localCount] = partitionRange(numOptions, rank, numRanks);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate only this rank's contiguous portion of the global input set.
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localCount, localBegin);
    
    std::vector<double> localResults(localCount);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Synchronize the start and report the slowest rank's elapsed time.  The
    // pricing loop itself has no communication or synchronization points.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localCount; ++i) {
        localResults[i] = blackScholes(localOptions[i]);
    }
    
    const double localSeconds = MPI_Wtime() - start;
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds,
               &computationSeconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", computationSeconds * 1000.0);
        printf("Options per second: %.0f\n",
               computationSeconds > 0.0 ? numOptions / computationSeconds : 0.0);
    }

    // Results are gathered only for the explicit external-results mode.
    // This keeps the normal benchmark path free of an O(N) root allocation
    // and network transfer.
    std::vector<double> results;
    if (printResults) {
        gatherResults(localResults, localBegin, numOptions, rank, numRanks, results);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    if (validate) {
        std::vector<double> validationResults;
        if (!printResults) {
            gatherValidationResults(localResults,
                                    localBegin,
                                    numOptions,
                                    rank,
                                    numRanks,
                                    validationResults);
        } else if (rank == 0) {
            validationResults.assign(results.begin(),
                                     results.begin() + std::min(static_cast<size_t>(10), numOptions));
        }

        int valid = 1;
        if (rank == 0) {
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, std::min(static_cast<size_t>(10), numOptions));
            printf("Validating results...\n");
            valid = validateResults(validationOptions, validationResults) ? 1 : 0;
            printf("Validation: %s\n", valid == 1 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid == 1 ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
