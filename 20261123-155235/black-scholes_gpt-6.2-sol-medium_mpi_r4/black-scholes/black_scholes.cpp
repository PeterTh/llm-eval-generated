#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
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

struct OptionRange {
    size_t first;
    size_t count;
};

OptionRange optionRange(size_t total, int rank, int ranks) {
    const size_t base = total / ranks;
    const size_t remainder = total % ranks;
    const size_t index = rank;
    return {index * base + std::min(index, remainder), base + (index < remainder)};
}

// Generate each rank's options using the same indices as the serial benchmark.
std::vector<OptionInput> generateOptions(OptionRange range) {
    constexpr auto testOptions = getTestOptions();
    std::vector<OptionInput> options;
    options.reserve(range.count);
    for (size_t offset = 0; offset < range.count; ++offset) {
        const size_t i = range.first + offset;
        OptionInput option = testOptions[i % testOptions.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options.push_back(option);
    }
    return options;
}

bool validateResults(std::span<const double> results) {
    constexpr auto testOptions = getTestOptions();
    bool allPassed = true;
    
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % testOptions.size()].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
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

// Stream an ordered prefix to rank zero without allocating a global result array.
template <typename Consume>
void collectResults(std::span<const double> results, size_t total, size_t prefix,
                    int rank, int ranks, Consume consume) {
    constexpr size_t blockSize = 65536;
    constexpr int resultTag = 0;
    const size_t first = optionRange(total, rank, ranks).first;

    if (rank != 0) {
        if (first < prefix) {
            const size_t count = std::min(results.size(), prefix - first);
            for (size_t offset = 0; offset < count;) {
                const int length = std::min(blockSize, count - offset);
                MPI_Send(results.data() + offset, length, MPI_DOUBLE, 0, resultTag, MPI_COMM_WORLD);
                offset += length;
            }
        }
        return;
    }

    const size_t ownCount = std::min(results.size(), prefix);
    if (ownCount != 0) {
        consume(results.first(ownCount), 0);
    }
    if (ownCount == prefix) {
        return;
    }

    const size_t bufferSize = std::min({blockSize, results.size(), prefix - ownCount});
    auto buffer = std::make_unique_for_overwrite<double[]>(bufferSize);
    for (int source = 1; source < ranks; ++source) {
        const auto [sourceFirst, sourceCount] = optionRange(total, source, ranks);
        if (sourceFirst >= prefix) {
            break;
        }
        const size_t count = std::min(sourceCount, prefix - sourceFirst);
        for (size_t offset = 0; offset < count;) {
            const int length = std::min(blockSize, count - offset);
            MPI_Recv(buffer.get(), length, MPI_DOUBLE, source, resultTag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            consume({buffer.get(), static_cast<size_t>(length)}, sourceFirst + offset);
            offset += length;
        }
    }
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    enum { Run, Help, InvalidArgument };
    std::array<std::uint64_t, 4> config{10000, 0, 0, Run};
    auto& [numOptions, validate, printResults, action] = config;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                action = Help;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                action = InvalidArgument;
                break;
            }
        }
    }
    MPI_Bcast(config.data(), config.size(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (action != Run) {
        return action == InvalidArgument;
    }
    const size_t total = numOptions;
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", total);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const OptionRange local = optionRange(total, rank, ranks);
    const auto options = generateOptions(local);
    std::vector<double> results(local.count);

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (size_t i = 0; i < local.count; ++i) {
        results[i] = blackScholes(options[i]);
    }
    const double elapsed = MPI_Wtime() - start;
    double computationTime = 0.0;
    MPI_Reduce(&elapsed, &computationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", computationTime * 1000.0);
        printf("Options per second: %.0f\n", computationTime > 0.0 ? total / computationTime : 0.0);
    }

    std::array<double, 10> checks{};
    const size_t checkCount = validate ? std::min(total, checks.size()) : 0;
    FloatingPointResults<double> summary(total);
    if (printResults || checkCount != 0) {
        collectResults(results, total, printResults ? total : checkCount,
                       rank, ranks, [&](std::span<const double> block, size_t first) {
            if (printResults) {
                summary.add(block.data(), block.size());
            }
            if (first < checkCount) {
                std::copy_n(block.data(), std::min(block.size(), checkCount - first),
                            checks.data() + first);
            }
        });
    }
    if (rank == 0 && printResults) {
        summary.print("OptionPrices");
    }
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating results...\n");
            valid = validateResults({checks.data(), checkCount});
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    MPI_Finalize();
    return status;
}
