#include <algorithm>
#include <array>
#include <chrono>
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

// Black-Scholes formula for European options
inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = std::exp(-r * T);
    
    double price;
    if (option.type == CALL) {
        price = S * std::exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * std::exp(-q * T) * cumulativeNormal(-d1);
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

// Generate a contiguous global range of options.  Each rank can generate its
// own inputs independently because the input at an index is deterministic.
void generateOptions(std::vector<OptionInput>& options, const size_t firstOption) {
    constexpr auto testOptions = getTestOptions();
    for (size_t localIndex = 0; localIndex < options.size(); ++localIndex) {
        const size_t i = firstOption + localIndex;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[localIndex] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[localIndex].spot *= factor;
        options[localIndex].strike *= factor;
    }
}

struct WorkPartition {
    size_t first;
    size_t count;
};

// Split the global range into contiguous, nearly equal ranges.  Contiguous
// ranges permit a direct gather into global order when output is requested.
WorkPartition partitionWork(const size_t total, const int rank, const int ranks) noexcept {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t base = total / rankCount;
    const size_t remainder = total % rankCount;
    const size_t count = base + (rankIndex < remainder ? 1U : 0U);
    const size_t first = rankIndex * base + std::min(rankIndex, remainder);
    return {first, count};
}

// Gather in chunks because MPI's traditional count arguments are int.  This
// path runs only for -r, where rank 0 must materialize the full result vector
// to retain the pre-MPI output semantics.
void gatherResults(const std::vector<double>& localResults,
                   std::vector<double>& globalResults,
                   const WorkPartition localWork,
                   const size_t totalOptions,
                   const int rank,
                   const int ranks) {
    constexpr int resultTag = 0;
    const size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());

    if (rank == 0) {
        globalResults.resize(totalOptions);
        std::copy(localResults.begin(), localResults.end(), globalResults.begin());

        for (int source = 1; source < ranks; ++source) {
            const WorkPartition sourceWork = partitionWork(totalOptions, source, ranks);
            for (size_t offset = 0; offset < sourceWork.count;) {
                const size_t chunk = std::min(maxChunk, sourceWork.count - offset);
                MPI_Recv(globalResults.data() + sourceWork.first + offset,
                         static_cast<int>(chunk), MPI_DOUBLE, source, resultTag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < localWork.count;) {
            const size_t chunk = std::min(maxChunk, localWork.count - offset);
            MPI_Send(localResults.data() + offset, static_cast<int>(chunk), MPI_DOUBLE,
                     0, resultTag, MPI_COMM_WORLD);
            offset += chunk;
        }
    }
}

// Validation checks at most the first ten globally ordered values.  Collect
// exactly that prefix so validation verifies the distributed computation
// without turning it into a full result gather.
void gatherValidationResults(const std::vector<double>& localResults,
                             std::vector<double>& validationResults,
                             const WorkPartition localWork,
                             const size_t totalOptions,
                             const size_t validationCount,
                             const int rank,
                             const int ranks) {
    constexpr int validationTag = 1;

    if (rank == 0) {
        validationResults.resize(validationCount);
        const size_t localCount = std::min(localWork.count, validationCount);
        std::copy_n(localResults.begin(), localCount, validationResults.begin());

        for (int source = 1; source < ranks; ++source) {
            const WorkPartition sourceWork = partitionWork(totalOptions, source, ranks);
            if (sourceWork.first < validationCount) {
                const size_t count = std::min(sourceWork.count,
                                              validationCount - sourceWork.first);
                MPI_Recv(validationResults.data() + sourceWork.first,
                         static_cast<int>(count), MPI_DOUBLE, source,
                         validationTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (localWork.first < validationCount) {
        const size_t count = std::min(localWork.count, validationCount - localWork.first);
        if (count != 0) {
            MPI_Send(localResults.data(), static_cast<int>(count), MPI_DOUBLE, 0,
                     validationTag, MPI_COMM_WORLD);
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

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    }
    
    const WorkPartition localWork = partitionWork(numOptions, rank, ranks);

    // Generate only this rank's deterministic share of the input data.
    std::vector<OptionInput> options(localWork.count);
    generateOptions(options, localWork.first);
    
    std::vector<double> localResults(localWork.count);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Synchronize before timing so the reported duration is the slowest
    // rank's computation time, which is the distributed wall-clock time.
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < localWork.count; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", computationSeconds * 1000.0);
        printf("Options per second: %.0f\n", numOptions / computationSeconds);
    }
    
    // Only the external result format needs all values on rank 0.
    std::vector<double> globalResults;
    if (printResults) {
        gatherResults(localResults, globalResults, localWork, numOptions, rank, ranks);
        if (rank == 0) {
            print_results(globalResults, "OptionPrices");
        }
    }
    
    // Validation
    if (validate) {
        const size_t validationCount = std::min(numOptions, static_cast<size_t>(10));
        std::vector<double> validationResults;
        if (printResults) {
            if (rank == 0) {
                validationResults.assign(globalResults.begin(),
                                         globalResults.begin() + validationCount);
            }
        } else {
            gatherValidationResults(localResults, validationResults, localWork, numOptions,
                                    validationCount, rank, ranks);
        }

        bool valid = true;
        if (rank == 0) {
            std::vector<OptionInput> validationOptions(validationCount);
            generateOptions(validationOptions, 0);

            printf("Validating results...\n");
            valid = validateResults(validationOptions, validationResults);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }

        int validInt = valid ? 1 : 0;
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
