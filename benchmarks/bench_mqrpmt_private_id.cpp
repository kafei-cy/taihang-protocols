/****************************************************************************
 * @file      bench_mqrpmt_private_id.cpp
 * @brief     Benchmark suite for mqRPMT-based Private-ID.
 * @author    Yang Cao
 *****************************************************************************/

#include <taihang/mpc/pso/mqrpmt_private_id.hpp>
#include <taihang/common/bench_setting.hpp>
#include <taihang/common/config.hpp>
#include <taihang/crypto/x25519_simd.hpp>

#include <openssl/obj_mac.h>

#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace taihang;
using namespace taihang::mpc;
namespace private_id = taihang::mpc::mqrpmt_private_id;

using Clock = std::chrono::high_resolution_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr int kBaseOtCurveId = NID_X9_62_prime256v1;
constexpr int kSecp256r1 = NID_X9_62_prime256v1;
constexpr size_t kLogItemLen = 16;
constexpr size_t kStatisticalSecurityParameter = 40;

struct BlockLess {
    bool operator()(const Block& lhs, const Block& rhs) const {
        return std::memcmp(&lhs, &rhs, sizeof(Block)) < 0;
    }
};

using BlockSet = std::set<Block, BlockLess>;

struct Dataset {
    std::vector<Block> vec_x;
    std::vector<Block> vec_y;
    size_t intersection_size = 0;
    size_t union_size = 0;
};

Dataset make_dataset(size_t item_len) {
    Dataset dataset;
    dataset.vec_x.resize(item_len);
    dataset.vec_y.resize(item_len);
    // Set the intersection size to be a half of the max item size.
    dataset.intersection_size = item_len / 2;
    dataset.union_size = item_len * 2 - dataset.intersection_size;

    for (size_t i = 0; i < item_len; ++i) {
        dataset.vec_y[i] = make_block(0x5959595959595959ULL,
                                      static_cast<uint64_t>(i + 1));
        if (i < dataset.intersection_size) {
            dataset.vec_x[i] = dataset.vec_y[i];
        } else {
            dataset.vec_x[i] = make_block(0x5858585858585858ULL,
                                          static_cast<uint64_t>(i + 1));
        }
    }
    return dataset;
}

struct BenchConfig {
    int mqrpmt_curve_id;
    uint16_t port;
    std::string label;
};

const std::vector<BenchConfig> kConfigs = {
    {kSecp256r1, 12660, "Private-ID / Secp256r1 / PlainSet"},
    {NID_X25519, 12661, "Private-ID / X25519 / PlainSet"},
};

private_id::PublicParameters make_public_parameters(const BenchConfig& config) {
    // Generate pp once; it must be shared by Sender and Receiver.
    return private_id::setup(kBaseOtCurveId,
                             config.mqrpmt_curve_id,
                             kLogItemLen,
                             kLogItemLen,
                             cwprf_mqrpmt::MembershipMode::PlainSet,
                             kStatisticalSecurityParameter);
}

struct BenchResult {
    double sender_ms = 0.0;
    double receiver_ms = 0.0;
    double wall_ms = 0.0;
    size_t sender_id_size = 0;
    size_t receiver_id_size = 0;
    size_t union_id_size = 0;
    bool passed = false;
};

bool validate_result(const Dataset& dataset,
                     const private_id::SenderOutput& sender_output,
                     const private_id::ReceiverOutput& receiver_output) {
    if (sender_output.sender_id.size() != dataset.vec_x.size() ||
        receiver_output.receiver_id.size() != dataset.vec_y.size()) {
        return false;
    }

    BlockSet expected_union(sender_output.sender_id.begin(), sender_output.sender_id.end());
    expected_union.insert(receiver_output.receiver_id.begin(), receiver_output.receiver_id.end());

    const BlockSet sender_union(sender_output.union_id.begin(), sender_output.union_id.end());
    const BlockSet receiver_union(receiver_output.union_id.begin(), receiver_output.union_id.end());

    return expected_union.size() == dataset.union_size &&
           sender_union == expected_union &&
           receiver_union == expected_union;
}

BenchResult run_once(const BenchConfig& config,
                     const private_id::PublicParameters& pp,
                     const Dataset& dataset) {
    const std::string address = "127.0.0.1";
    BenchResult result;

    const auto wall_begin = Clock::now();

    auto sender_future = std::async(std::launch::async, [&] {
        net::NetIO io("server", address, config.port);
        const auto begin = Clock::now();
        private_id::SenderOutput output = private_id::sender(io, pp, dataset.vec_x);
        return std::make_pair(std::move(output), Milliseconds(Clock::now() - begin).count());
    });

    auto receiver_future = std::async(std::launch::async, [&] {
        net::NetIO io("client", address, config.port);
        const auto begin = Clock::now();
        private_id::ReceiverOutput output = private_id::receiver(io, pp, dataset.vec_y);
        return std::make_pair(std::move(output), Milliseconds(Clock::now() - begin).count());
    });

    auto sender_pair = sender_future.get();
    auto receiver_pair = receiver_future.get();

    result.wall_ms = Milliseconds(Clock::now() - wall_begin).count();
    result.sender_ms = sender_pair.second;
    result.receiver_ms = receiver_pair.second;
    result.sender_id_size = sender_pair.first.sender_id.size();
    result.receiver_id_size = receiver_pair.first.receiver_id.size();
    result.union_id_size = receiver_pair.first.union_id.size();
    // Receiver sends the shuffled union_id set back to Sender; 
    result.passed = validate_result(dataset, sender_pair.first, receiver_pair.first);
    return result;
}

double kelem_per_second(size_t element_count, double milliseconds) {
    return static_cast<double>(element_count) / milliseconds;
}

void print_header(const BenchConfig& config, size_t item_len) {
    std::cout
        << "\n"
        << "================================================================================\n"
        << config.label << "\n"
        << "--------------------------------------------------------------------------------\n"
        << "Sender set   (X): 2^" << kLogItemLen << " = " << item_len << "\n"
        << "Receiver set (Y): 2^" << kLogItemLen << " = " << item_len << "\n"
        << "Intersection    : 50%\n";
    if (config.mqrpmt_curve_id == NID_X25519) {
        std::cout << "X25519 Backend  : "
                  << crypto::x25519_simd::backend_name(crypto::x25519_simd::Backend::Auto)
                  << "\n";
    }
    std::cout << "================================================================================\n";
}

void print_result(const Dataset& dataset, const BenchResult& result) {
    std::cout << std::fixed << std::setprecision(2)
              << "Wall time       : " << result.wall_ms / 1000.0 << " s\n"
              << "Sender time     : " << result.sender_ms / 1000.0 << " s ("
              << kelem_per_second(dataset.vec_x.size(), result.sender_ms)
              << " Kelem/s)\n"
              << "Receiver time   : " << result.receiver_ms / 1000.0 << " s ("
              << kelem_per_second(dataset.vec_y.size(), result.receiver_ms)
              << " Kelem/s)\n"
              << "Sender ID size  : " << result.sender_id_size << "\n"
              << "Receiver ID size: " << result.receiver_id_size << "\n"
              << "Union ID size   : " << result.union_id_size
              << " (expected " << dataset.union_size << ")\n"
              << "Validation      : " << (result.passed ? "PASS" : "FAIL") << "\n";
}

void print_summary(const std::vector<std::pair<BenchConfig, BenchResult>>& results,
                   size_t item_len) {
    constexpr int kLabelWidth = 38;

    std::cout
        << "\n\n"
        << "====================================================================================================\n"
        << "                            Taihang mqRPMT Private-ID Benchmark Summary\n"
        << "====================================================================================================\n"
        << "Dataset: 2^" << kLogItemLen << " (" << item_len << ") elements per party, 50% intersection\n"
        << "Threads: " << config::thread_num << " per party ("
        << config::thread_num * 2 << " concurrent total on single-machine loopback)\n"
        << "----------------------------------------------------------------------------------------------------\n"
        << std::left << std::setw(kLabelWidth) << "Configuration"
        << std::right << std::setw(12) << "Sender(s)"
        << std::setw(13) << "Receiver(s)"
        << std::setw(11) << "Wall(s)"
        << std::setw(14) << "Kelem/s"
        << std::setw(10) << "Status"
        << "\n"
        << "----------------------------------------------------------------------------------------------------\n";

    for (const auto& [config, result] : results) {
        std::cout << std::left << std::setw(kLabelWidth)
                  << config.label << std::right << std::fixed
                  << std::setprecision(2)
                  << std::setw(12) << result.sender_ms / 1000.0
                  << std::setw(13) << result.receiver_ms / 1000.0
                  << std::setw(11) << result.wall_ms / 1000.0
                  << std::setw(14) << kelem_per_second(item_len, result.wall_ms)
                  << std::setw(10) << (result.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout
        << "====================================================================================================\n";
}

int main() {
    const size_t item_len = size_t{1} << kLogItemLen;

    thread_configuration(BenchmarkMode::SingleMachine);

    // Generate testcase.
    std::cout << "Generating benchmark dataset...\n";
    const auto dataset_begin = Clock::now();
    const Dataset dataset = make_dataset(item_len);
    std::cout << "Dataset ready in " << std::fixed << std::setprecision(2)
              << Milliseconds(Clock::now() - dataset_begin).count()
              << " ms\n";

    std::vector<std::pair<BenchConfig, BenchResult>> summary;
    summary.reserve(kConfigs.size());

    for (const BenchConfig& config : kConfigs) {
        print_header(config, item_len);
        const auto pp = make_public_parameters(config);
        const BenchResult result = run_once(config, pp, dataset);
        print_result(dataset, result);
        summary.emplace_back(config, result);
    }

    print_summary(summary, item_len);
    return 0;
}
