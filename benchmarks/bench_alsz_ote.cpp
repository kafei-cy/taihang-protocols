/****************************************************************************
 * @file      bench_alsz_ote.cpp
 * @brief     Computation and communication benchmark for ALSZ OT extension.
 * @details   Measures complete chosen-message OT batches with fresh base OTs.
 * @author    Yu Chen
 *****************************************************************************/

#include <taihang/common/config.hpp>
#include <taihang/mpc/ot/alsz_ote.hpp>

#include <openssl/obj_mac.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace taihang;
using namespace taihang::mpc::alsz_ote;

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr size_t kBaseOtLen = 128;
constexpr int kCurveNid = NID_X9_62_prime256v1;
constexpr char kAddress[] = "127.0.0.1";
constexpr uint64_t kInputSeed = 0x243f6a8885a308d3ULL;

struct Options {
    size_t count = 1ULL << 20;
    int threads = 1;
    size_t repeat = 5;
    size_t warmup = 1;
    uint16_t port = 12347;
};

struct BenchmarkInputs {
    std::vector<Block> messages_0;
    std::vector<Block> messages_1;
    std::vector<uint8_t> choices;
};

struct CommunicationCost {
    uint64_t base_ot_bytes;
    uint64_t extension_bytes;
    uint64_t sender_to_receiver_bytes;
    uint64_t receiver_to_sender_bytes;

    uint64_t total_bytes() const {
        return sender_to_receiver_bytes + receiver_to_sender_bytes;
    }
};

struct PartyTiming {
    double elapsed_ms;
    Clock::time_point end;
};

struct BenchmarkResult {
    double sender_ms;
    double receiver_ms;
    double wall_ms;
    bool verified;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--count N] [--threads T] [--repeat R] [--warmup W] [--port P]\n"
              << "  --count    Actual OT count, a multiple of 128 in [128, 67108864]\n"
              << "             (default: 1048576)\n"
              << "  --threads  OpenMP threads per party in [1, 256] (default: 1)\n"
              << "  --repeat   Measured batches in [1, 1000] (default: 5)\n"
              << "  --warmup   Unmeasured batches in [0, 1000] (default: 1)\n"
              << "  --port     Loopback TCP port in [1024, 65535] (default: 12347)\n";
}

uint64_t parse_number(std::string_view text, const std::string& option) {
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Invalid number for " + option);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option != "--count" && option != "--threads" && option != "--repeat"
            && option != "--warmup" && option != "--port") {
            throw std::invalid_argument("Unknown option: " + option);
        }
        if (++i == argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const uint64_t value = parse_number(argv[i], option);
        if (option == "--count") {
            if (value < 128 || value > (1ULL << 26) || value % 128 != 0) {
                throw std::invalid_argument("--count must be a multiple of 128 in [128, 67108864]");
            }
            options.count = static_cast<size_t>(value);
        } else if (option == "--threads") {
            if (value < 1 || value > 256) {
                throw std::invalid_argument("--threads must be in [1, 256]");
            }
            options.threads = static_cast<int>(value);
        } else if (option == "--repeat") {
            if (value < 1 || value > 1000) {
                throw std::invalid_argument("--repeat must be in [1, 1000]");
            }
            options.repeat = static_cast<size_t>(value);
        } else if (option == "--warmup") {
            if (value > 1000) {
                throw std::invalid_argument("--warmup must be in [0, 1000]");
            }
            options.warmup = static_cast<size_t>(value);
        } else {
            if (value < 1024 || value > 65535) {
                throw std::invalid_argument("--port must be in [1024, 65535]");
            }
            options.port = static_cast<uint16_t>(value);
        }
    }
    return options;
}

// Reproducible public benchmark inputs; protocol randomness remains unchanged.
uint64_t splitmix64(uint64_t& state) {
    uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

BenchmarkInputs make_inputs(size_t count) {
    BenchmarkInputs inputs{
        std::vector<Block>(count),
        std::vector<Block>(count),
        std::vector<uint8_t>(count),
    };
    uint64_t state = kInputSeed;
    for (size_t i = 0; i < count; ++i) {
        // Keep this order identical to the EMP comparison benchmark.
        const uint64_t m0_low = splitmix64(state);
        const uint64_t m0_high = splitmix64(state);
        const uint64_t m1_low = splitmix64(state);
        const uint64_t m1_high = splitmix64(state);
        inputs.messages_0[i] = make_block(m0_high, m0_low);
        inputs.messages_1[i] = make_block(m1_high, m1_low);
        inputs.choices[i] = static_cast<uint8_t>(splitmix64(state) & 1);
    }
    return inputs;
}

CommunicationCost communication_cost(size_t ot_count,
                                     size_t base_ot_count,
                                     size_t point_bytes) {
    const uint64_t n = ot_count;
    const uint64_t k = base_ot_count;
    const uint64_t block_bytes = sizeof(Block);

    // The ALSZ sender acts as the receiver in the Naor-Pinkas base OTs.
    const uint64_t base_sender_to_receiver = point_bytes * k;
    const uint64_t base_receiver_to_sender =
        point_bytes * (k + 1) + 2 * k * block_bytes;

    const uint64_t ciphertexts = 2 * n * block_bytes;
    const uint64_t adjustment_matrix = n * k / 8;

    return {
        base_sender_to_receiver + base_receiver_to_sender,
        ciphertexts + adjustment_matrix,
        base_sender_to_receiver + ciphertexts,
        base_receiver_to_sender + adjustment_matrix,
    };
}

BenchmarkResult run_batch(const PublicParameters& pp, const Options& options,
                          const BenchmarkInputs& inputs) {
    std::vector<Block> received;
    std::latch parties_ready(2);
    std::latch start_protocol(1);

    auto sender_future = std::async(std::launch::async, [&]() {
        net::NetIO io("server", kAddress, options.port);
        parties_ready.count_down();
        start_protocol.wait();

        const auto begin = Clock::now();
        sender<BlockPolicy>(io, pp, inputs.messages_0, inputs.messages_1,
                            options.count);
        io.flush();
        const auto end = Clock::now();
        return PartyTiming{Milliseconds(end - begin).count(), end};
    });

    auto receiver_future = std::async(std::launch::async, [&]() {
        net::NetIO io("client", kAddress, options.port);
        parties_ready.count_down();
        start_protocol.wait();

        const auto begin = Clock::now();
        received = receiver<BlockPolicy>(io, pp, inputs.choices, options.count);
        io.flush();
        const auto end = Clock::now();
        return PartyTiming{Milliseconds(end - begin).count(), end};
    });

    parties_ready.wait();
    const auto wall_begin = Clock::now();
    start_protocol.count_down();

    const PartyTiming sender_timing = sender_future.get();
    const PartyTiming receiver_timing = receiver_future.get();
    // End at protocol completion, excluding futures and NetIO destruction.
    const auto wall_end = std::max(sender_timing.end, receiver_timing.end);
    const double wall_ms = Milliseconds(wall_end - wall_begin).count();

    bool verified = received.size() == options.count;
    for (size_t i = 0; verified && i < options.count; ++i) {
        const Block& expected = inputs.choices[i] == 0
            ? inputs.messages_0[i] : inputs.messages_1[i];
        verified = received[i] == expected;
        if (!verified) {
            std::cerr << "OT verification failed at index " << i << '\n';
        }
    }

    return {sender_timing.elapsed_ms, receiver_timing.elapsed_ms, wall_ms, verified};
}

void print_configuration(const PublicParameters& pp, const Options& options) {
    const size_t point_bytes = pp.base_ot_pp.group_ctx->get_point_byte_len();
    const CommunicationCost cost = communication_cost(options.count, pp.base_len, point_bytes);
    std::cout
        << "\n================================================================================\n"
        << "Taihang ALSZ OT Extension Benchmark\n"
        << "--------------------------------------------------------------------------------\n"
        << "Mode                  : single-machine loopback\n"
        << "Protocol              : semi-honest, complete chosen-message 1-out-of-2 OT\n"
        << "Message size          : 128 bits\n"
        << "Base OTs              : " << pp.base_len << " Naor-Pinkas OTs on NIST P-256\n"
        << "EC point encoding     : " << point_bytes << " bytes (compressed)\n"
        << "OpenMP threads/party  : " << options.threads << '\n'
        << "Actual OT count       : " << options.count << '\n'
        << "Warmup / repetitions  : " << options.warmup << " / " << options.repeat << '\n'
        << "Base OT policy        : fresh base OTs inside every timed batch, including warmup\n"
        << "Timing scope          : full OT and final flush; inputs, connection setup,\n"
        << "                        verification and socket destruction excluded\n"
        << "Input seed            : 0x243f6a8885a308d3 (SplitMix64, public test inputs only)\n"
        << "Estimated base bytes  : " << cost.base_ot_bytes << '\n'
        << "Estimated OTE bytes   : " << cost.extension_bytes << '\n'
        << "Estimated S -> R bytes: " << cost.sender_to_receiver_bytes << '\n'
        << "Estimated R -> S bytes: " << cost.receiver_to_sender_bytes << '\n'
        << "Estimated total bytes : " << cost.total_bytes() << '\n'
        << "Communication scope   : formula-derived application payload, excluding TCP/IP\n"
        << "================================================================================\n"
        << "# sample,implementation,round,count,threads,sender_ms,receiver_ms,wall_ms,verified\n";
}

void print_sample(const char* kind, size_t round, const Options& options,
                  const BenchmarkResult& result) {
    std::cout << kind << ",taihang," << round << ',' << options.count << ','
              << options.threads << ',' << std::fixed << std::setprecision(6)
              << result.sender_ms << ',' << result.receiver_ms << ','
              << result.wall_ms << ',' << (result.verified ? 1 : 0) << '\n';
}

void print_summary(const char* metric, std::vector<double> values,
                   const Options& options) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    const double median = values.size() % 2 == 0
        ? (values[mid - 1] + values[mid]) / 2 : values[mid];
    std::cout << "summary,taihang," << metric << ',' << options.count << ','
              << options.threads << ',' << options.repeat << ','
              << std::fixed << std::setprecision(6) << median << ','
              << values.front() << ',' << values.back() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help"
                          || std::string_view(argv[1]) == "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        const Options options = parse_options(argc, argv);
        config::use_point_compression = true;
        config::thread_num = options.threads;
        const PublicParameters pp = setup(kCurveNid, kBaseOtLen);
        const BenchmarkInputs inputs = make_inputs(options.count);
        print_configuration(pp, options);

        // Warmup initializes machine/library state without reusing base-OT seeds.
        for (size_t round = 0; round < options.warmup; ++round) {
            const BenchmarkResult result = run_batch(pp, options, inputs);
            print_sample("warmup", round + 1, options, result);
            if (!result.verified) return 1;
        }

        std::vector<double> sender_times, receiver_times, wall_times;
        for (size_t round = 0; round < options.repeat; ++round) {
            const BenchmarkResult result = run_batch(pp, options, inputs);
            print_sample("sample", round + 1, options, result);
            if (!result.verified) return 1;
            sender_times.push_back(result.sender_ms);
            receiver_times.push_back(result.receiver_ms);
            wall_times.push_back(result.wall_ms);
        }

        std::cout << "# summary,implementation,metric,count,threads,repeat,median_ms,min_ms,max_ms\n";
        print_summary("sender", sender_times, options);
        print_summary("receiver", receiver_times, options);
        print_summary("wall", wall_times, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
