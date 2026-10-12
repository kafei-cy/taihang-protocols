/****************************************************************************
 * @file      cwprf_mqrpmt.cpp
 * @brief     Implementation of weak commutative PRF multi-query RPMT.
 * @author    This file is part of Taihang, developed by Yu Chen.
 *****************************************************************************/

#include <taihang/mpc/pso/cwprf_mqrpmt.hpp>
#include <taihang/crypto/x25519_simd.hpp>
#include <taihang/common/logger.hpp>
#include <algorithm> // std::shuffle
#include <format>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <unordered_set>
#include <random>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>

namespace taihang::mpc::cwprf_mqrpmt {

namespace x25519_simd = taihang::crypto::x25519_simd;

// -------------------------------------------------------------------------
// PublicParameters helpers
// -------------------------------------------------------------------------

std::string PublicParameters::format() const {
    std::ostringstream oss;
    oss << "[cwPRF-based mqRPMT PublicParameters]\n";
    oss << "Curve ID               :" << curve_id << "\n";
    oss << "log2(server set size)  :" << log_server_len << "\n";
    oss << "log2(client set size)  :" << log_client_len << "\n";
    oss << "Membership mode        :"
        << (membership_mode == MembershipMode::BloomFilter ? "BloomFilter" : "PlainSet") << "\n";
    if (membership_mode == MembershipMode::BloomFilter) {
        oss << "  Statistical security parameter : " << statistical_security_parameter << "\n";
    }
    return oss.str();
}

// Serialisation format:
// curve_id  log_server_len  log_client_len  membership_mode ssp 
// membership_mode is stored as its underlying integer (0 = BloomFilter, 1 = PlainSet).
std::ostream& operator<<(std::ostream& os, const PublicParameters& pp) {
    os << pp.curve_id                        << " "
       << pp.log_server_len                  << " "
       << pp.log_client_len                  << " "
       << static_cast<int>(pp.membership_mode)   << " "
       << pp.statistical_security_parameter; 

    return os;
}

std::istream& operator>>(std::istream& is, PublicParameters& pp) {
    int mode_int = 0;
    is >> pp.curve_id
       >> pp.log_server_len
       >> pp.log_client_len
       >> mode_int 
       >> pp.statistical_security_parameter;

    pp.membership_mode = static_cast<MembershipMode>(mode_int);

    // Reconstruct contexts with stable heap addresses
    if(pp.curve_id != NID_X25519){
        pp.group_ctx = std::make_shared<ECGroup>(pp.curve_id);
        pp.ring_ctx = std::make_shared<Zn>(pp.group_ctx->order);
    }
    else{
        pp.group_ctx = nullptr;
        pp.ring_ctx = nullptr;
    }    
    return is;
}

// -------------------------------------------------------------------------
// setup
// -------------------------------------------------------------------------

PublicParameters setup(int curve_id, 
                       size_t log_server_len, 
                       size_t log_client_len, 
                       MembershipMode mode, 
                       std::optional<size_t> statistical_security_parameter) {
    PublicParameters pp;
    pp.curve_id = curve_id;
    // Allocate contexts with stable addresses
    if(curve_id != NID_X25519){
        pp.group_ctx = std::make_shared<ECGroup>(pp.curve_id);
        pp.ring_ctx = std::make_shared<Zn>(pp.group_ctx->order); 
    }
    else{
        pp.group_ctx = nullptr;
        pp.ring_ctx = nullptr; 
    }
    pp.log_server_len = log_server_len;
    pp.log_client_len = log_client_len;

    pp.membership_mode    = mode; 

    if (mode == MembershipMode::BloomFilter) {
        TAIHANG_ASSERT(statistical_security_parameter.has_value(), "BloomFilter mode requires statistical_security_parameter.");
        pp.statistical_security_parameter = statistical_security_parameter.value();
    } 
    else {
        pp.statistical_security_parameter = 0; // plain_set mode
    }

    return pp;
}

std::vector<uint8_t> server(net::NetIO& io, const PublicParameters& pp, const std::vector<Block>& vec_y) {
    TAIHANG_TIMER("cwPRF mqRPMT:", "Server total execution time");
    TAIHANG_LOG("cwPRF mqRPMT:", "Server protocol context initiated >>>");

    size_t server_len = static_cast<size_t>(std::pow(2, pp.log_server_len));
    size_t client_len = static_cast<size_t>(std::pow(2, pp.log_client_len));

    // result
    std::vector<uint8_t> vec_indication_bit(client_len);

    TAIHANG_ASSERT(server_len == vec_y.size(), "cwPRF mqRPMT: Server input size mismatch.");

    if(pp.curve_id != NID_X25519){        
        // Pick a secret random exponent key k1 from the Zn scalar field
        ZnElement k1 = pp.ring_ctx->gen_random();
    
        // Step 1: Compute F_k1(H(y_i)) = H(y_i)^k1
        std::vector<ECPoint> vec_fk1_y(server_len, pp.group_ctx);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < server_len; ++i) {
            vec_fk1_y[i] = hash_to_curve_fast(vec_y[i], *pp.group_ctx) * k1;
        }

        TAIHANG_LOG("cwPRF mqRPMT [step 1]:", std::format("Server ===> F_k1(H(y_i)) ===> Client [{:.2f} MB]", 
                                          static_cast<double>(server_len * point_byte_len) / (1024 * 1024)));
        // Send F_k1(y_i) elements immediately to client
        io.send(vec_fk1_y);

        TAIHANG_LOG("cwPRF mqRPMT [step 2]:", "Server receives F_k2(H(x_i)) from Client...");
        // Receive F_k2(x_i) from client
        std::vector<ECPoint> vec_fk2_x(client_len, pp.group_ctx);
        io.recv(vec_fk2_x);

        // Step 2: Compute commutative composite layer F_k1k2(x_i) = (F_k2(x_i))^k1
        std::vector<ECPoint> vec_fk1k2_x(client_len, pp.group_ctx);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < client_len; ++i) {
            vec_fk1k2_x[i] = vec_fk2_x[i] * k1;
        }

        // Step 2: membership test (mode-dependent)
        if (pp.membership_mode == MembershipMode::BloomFilter) {
            TAIHANG_LOG("cwPRF mqRPMT [step 2]:", "Server receives bloom filter from Client...");
            // Receive Bloom Filter payload metadata and mapping from client
            size_t filter_size = 0;
            io.recv(filter_size);
            std::vector<char> buffer(filter_size);
            io.recv(buffer.data(), filter_size);

            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Server reconstructing Bloom Filter structure [{:.2f} MB]", 
                                                          static_cast<double>(filter_size) / (1024 * 1024)));
            // Reconstruct Bloom Filter representation object
            BloomFilter filter;
            bool deserialize_success = filter.deserialize(buffer.data());
            TAIHANG_ASSERT(deserialize_success, "cwPRF mqRPMT: Bloom Filter parsing failed.");

            TAIHANG_LOG("cwPRF mqRPMT [step 4]:", "Server running batched membership lookup over intersection domain");
            // Determine intersection membership representation
            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < client_len; ++i) {
                auto bytes = vec_fk1k2_x[i].to_bytes();
                vec_indication_bit[i] = filter.contains(bytes.data(), bytes.size()) ? 1 : 0;
            }
        }
        else{
            // Receive the shuffled F_k2k1(y_i) set
            std::vector<ECPoint> vec_fk2k1_y(server_len, pp.group_ctx);
            io.recv(vec_fk2k1_y);

            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Server received shuffled PlainSet [{:.2f} MB]",
                        static_cast<double>(server_len * pp.group_ctx->get_point_byte_len()) / (1024 * 1024)));

            // Build lookup set (byte-string keyed)
            std::unordered_set<ECPoint, ECPointHash> plain_set;
            plain_set.reserve(server_len);
            for (size_t i = 0; i < server_len; ++i) {
                plain_set.insert(vec_fk2k1_y[i]); 
            }

            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < client_len; ++i) {
                vec_indication_bit[i] = plain_set.contains(vec_fk1k2_x[i]) ? 1 : 0;
            }
        }
    }
    else{
        // Pick a secret random exponent key k1
        std::vector<uint8_t> k1(EC25519Point::SCALAR_BYTE_LEN); 
        TAIHANG_CHECK(RAND_bytes(k1.data(), EC25519Point::SCALAR_BYTE_LEN) == 1, "RAND_bytes failed");  
  
        // Step 1: Compute F_k1(H(y_i)) = H(y_i)^k1
        std::vector<EC25519Point> vec_fk1_y(server_len);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < server_len; ++i) {
            vec_fk1_y[i] = hash_to_curve25519(vec_y[i]);
        }
        x25519_simd::scalar_mul_batch(vec_fk1_y, k1);

        TAIHANG_LOG("cwPRF mqRPMT [step 1]:", std::format("Server ===> F_k1(H(y_i)) ===> Client [{:.2f} MB]", 
                            static_cast<double>(server_len * EC25519Point::POINT_BYTE_LEN) / (1024 * 1024)));
        // Send F_k1(y_i) elements immediately to client
        io.send(vec_fk1_y);

        TAIHANG_LOG("cwPRF mqRPMT [step 2]:", "Server receives F_k2(H(x_i)) from Client...");
        // Receive F_k2(x_i) from client
        std::vector<EC25519Point> vec_fk1k2_x(client_len);
        io.recv(vec_fk1k2_x);

        // Step 2: Compute commutative composite layer F_k1k2(x_i) = (F_k2(x_i))^k1
        x25519_simd::scalar_mul_batch(vec_fk1k2_x, k1);

        // Step 2: membership test (mode-dependent)
        if (pp.membership_mode == MembershipMode::BloomFilter) {
            TAIHANG_LOG("cwPRF mqRPMT [step 2]:", "Server receives bloom filter from Client...");
            // Receive Bloom Filter payload metadata and mapping from client
            size_t filter_size = 0;
            io.recv(filter_size);
            std::vector<char> buffer(filter_size);
            io.recv(buffer.data(), filter_size);

            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Server reconstructing Bloom Filter structure [{:.2f} MB]", 
                                                  static_cast<double>(filter_size) / (1024 * 1024)));
            // Reconstruct Bloom Filter representation object
            BloomFilter filter;
            bool deserialize_success = filter.deserialize(buffer.data());
            TAIHANG_ASSERT(deserialize_success, "cwPRF mqRPMT: Bloom Filter parsing failed.");

            TAIHANG_LOG("cwPRF mqRPMT [step 4]:", "Server running batched membership lookup over intersection domain");
            // Determine intersection membership representation
            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < client_len; ++i) {
                vec_indication_bit[i] =
                    filter.contains(vec_fk1k2_x[i].px, EC25519Point::POINT_BYTE_LEN) ? 1 : 0;
            }
        }
        else{
            // Receive the shuffled F_k2k1(y_i) set
            std::vector<EC25519Point> vec_fk2k1_y(server_len);
            io.recv(vec_fk2k1_y);

            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Server received shuffled PlainSet [{:.2f} MB]",
                        static_cast<double>(server_len * EC25519Point::POINT_BYTE_LEN) / (1024 * 1024)));

            // Build lookup set (byte-string keyed)
            std::unordered_set<EC25519Point, EC25519PointHash> plain_set;
            plain_set.reserve(server_len);
            for (size_t i = 0; i < server_len; ++i) {
                plain_set.insert(vec_fk2k1_y[i]); 
            }

            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < client_len; ++i) {
                vec_indication_bit[i] = plain_set.contains(vec_fk1k2_x[i]) ? 1 : 0;
            }
        }
    }

    TAIHANG_LOG("cwPRF mqRPMT:", "Server protocol context successfully completed <<<");
    return vec_indication_bit;
}

void client(net::NetIO& io, const PublicParameters& pp, const std::vector<Block>& vec_x) {
    TAIHANG_TIMER("cwPRF mqRPMT:", "Client total execution time");
    TAIHANG_LOG("cwPRF mqRPMT:", "Client protocol context initiated >>>");

    size_t server_len = static_cast<size_t>(std::pow(2, pp.log_server_len));
    size_t client_len = static_cast<size_t>(std::pow(2, pp.log_client_len));

    TAIHANG_ASSERT(client_len == vec_x.size(), "cwPRF mqRPMT: Client input size mismatch.");

    if(pp.curve_id != NID_X25519){  
        // Pick a secret random exponent key k2 from the Zn scalar field
        ZnElement k2 = pp.ring_ctx -> gen_random();

        // Step 1: Compute F_k2(H(x_i)) = H(x_i)^k2
        std::vector<ECPoint> vec_fk2_x(client_len, pp.group_ctx);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < client_len; ++i) {
            vec_fk2_x[i] = hash_to_curve_fast(vec_x[i], *pp.group_ctx) * k2;
        }

        TAIHANG_LOG("cwPRF mqRPMT [step 1]:", "Client receives F_k1(H(y_i)) from Server...");
        // Receive incoming server layers F_k1(y_i)
        std::vector<ECPoint> vec_fk1_y(server_len);
        io.recv(vec_fk1_y);

        TAIHANG_LOG("cwPRF mqRPMT [step 2]:", std::format("Client ===> F_k2(H(x_i)) ===> Server [{:.2f} MB]", 
                                        static_cast<double>(client_len * pp.group_ctx->get_point_byte_len()) / (1024 * 1024)));
        // Send blindings out immediately
        io.send(vec_fk2_x);

        // Step 2: Compute commutative PRF values F_k2k1(y_i) = (F_k1(y_i))^k2
        std::vector<ECPoint> vec_fk2k1_y(server_len, pp.group_ctx);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < server_len; ++i) {
            vec_fk2k1_y[i] = vec_fk1_y[i] * k2;
        }

        // send membership structure (mode-dependent)
        if (pp.membership_mode == MembershipMode::BloomFilter) {
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", "Client builds and sends Bloom Filter");
            // Step 3: Initialize and populate Bloom Filter structure with F_k2k1(y_i)
            BloomFilter filter(vec_fk2k1_y.size(), pp.statistical_security_parameter);

            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < server_len; ++i) {
                auto bytes = vec_fk2k1_y[i].to_bytes();
                filter.insert(bytes.data(), bytes.size());
            }

            // Serialize and ship the Bloom Filter structure across the wire
            size_t filter_size = filter.get_serialized_size();
            io.send(filter_size);

            std::vector<char> buffer(filter_size);
            bool serialize_success = filter.serialize(buffer.data());
            TAIHANG_ASSERT(serialize_success, "cwPRF mqRPMT: Bloom Filter serialization failed.");
    
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Client ===> BloomFilter(F_k2k1(y_i)) ===> Server [{:.2f} MB]", 
                                                  static_cast<double>(filter_size) / (1024 * 1024)));
            io.send(buffer.data(), filter_size);
        }
        else{
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", "Client shuffling and sending PlainSet...");

            // Shuffle to hide correspondence with server's original ordering
            thread_local std::mt19937_64 rng{std::random_device{}()};
            std::shuffle(vec_fk2k1_y.begin(), vec_fk2k1_y.end(), rng);
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Client ===> Permuted PlainSet(F_k2k1(y_i)) ===> Server [{:.2f} MB]",
                                    static_cast<double>(server_len * pp.group_ctx->get_point_byte_len()) / (1024 * 1024)));
            io.send(vec_fk2k1_y); 
        }
    }
    else{
        // Pick a secret random exponent key k2
        std::vector<uint8_t> k2(EC25519Point::SCALAR_BYTE_LEN); 
        TAIHANG_CHECK(RAND_bytes(k2.data(), EC25519Point::SCALAR_BYTE_LEN) == 1, "RAND_bytes failed");  

        // Step 1: Compute F_k2(H(x_i)) = H(x_i)^k2
        std::vector<EC25519Point> vec_fk2_x(client_len);
        #pragma omp parallel for num_threads(config::thread_num)
        for (size_t i = 0; i < client_len; ++i) {
            vec_fk2_x[i] = hash_to_curve25519(vec_x[i]);
        }
        x25519_simd::scalar_mul_batch(vec_fk2_x, k2);

        TAIHANG_LOG("cwPRF mqRPMT [step 1]:", "Client receives F_k1(H(y_i)) from Server...");
        // Receive incoming server layers F_k1(y_i)
        std::vector<EC25519Point> vec_fk2k1_y(server_len);
        io.recv(vec_fk2k1_y);

        TAIHANG_LOG("cwPRF mqRPMT [step 2]:", std::format("Client ===> F_k2(H(x_i)) ===> Server [{:.2f} MB]", 
                                        static_cast<double>(client_len * EC25519Point::POINT_BYTE_LEN) / (1024 * 1024)));
        // Send blindings out immediately
        io.send(vec_fk2_x);

        // Step 2: Compute commutative PRF values F_k2k1(y_i) = (F_k1(y_i))^k2
        x25519_simd::scalar_mul_batch(vec_fk2k1_y, k2);

        // send membership structure (mode-dependent)
        if (pp.membership_mode == MembershipMode::BloomFilter) {
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", "Client builds and sends Bloom Filter");
            // Step 3: Initialize and populate Bloom Filter structure with F_k2k1(y_i)
            BloomFilter filter(vec_fk2k1_y.size(), pp.statistical_security_parameter);

            #pragma omp parallel for num_threads(config::thread_num)
            for (size_t i = 0; i < server_len; ++i) {
                // auto bytes = vec_fk2k1_y[i].to_bytes();
                // filter.insert(bytes.data(), bytes.size());
                filter.insert(vec_fk2k1_y[i].px, EC25519Point::POINT_BYTE_LEN);
            }

            // Serialize and ship the Bloom Filter structure across the wire
            size_t filter_size = filter.get_serialized_size();
            io.send(filter_size);

            std::vector<char> buffer(filter_size);
            bool serialize_success = filter.serialize(buffer.data());
            TAIHANG_ASSERT(serialize_success, "cwPRF mqRPMT: Bloom Filter serialization failed.");
    
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Client ===> BloomFilter(F_k2k1(y_i)) ===> Server [{:.2f} MB]", 
                                                  static_cast<double>(filter_size) / (1024 * 1024)));
            io.send(buffer.data(), filter_size);
        }
        else{
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", "Client shuffling and sending PlainSet...");

            // Shuffle to hide correspondence with server's original ordering
            thread_local std::mt19937_64 rng{std::random_device{}()};
            std::shuffle(vec_fk2k1_y.begin(), vec_fk2k1_y.end(), rng);
            TAIHANG_LOG("cwPRF mqRPMT [step 3]:", std::format("Client ===> Permuted PlainSet(F_k2k1(y_i)) ===> Server [{:.2f} MB]",
                                    static_cast<double>(server_len * EC25519Point::POINT_BYTE_LEN) / (1024 * 1024)));
            io.send(vec_fk2k1_y); 
        }
    }

    TAIHANG_LOG("cwPRF mqRPMT:", "Client protocol context successfully completed <<<");
}

} // namespace taihang::mpc::cwprf_mqrpmt
