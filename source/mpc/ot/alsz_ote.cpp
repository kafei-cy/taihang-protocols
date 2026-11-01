/****************************
 * @file      alsz_ote.cpp
 * @brief     Implementation of ALSZ OT Extension.
 ****************************/

#include <taihang/mpc/ot/alsz_ote.hpp>
#include <taihang/common/check.hpp>
#include <taihang/common/logger.hpp>
#include <taihang/crypto/prg.hpp>
#include <taihang/crypto/stream_cipher.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <omp.h>
#include <type_traits>
#include <utility>

namespace taihang::mpc::alsz_ote {

#if defined(TAIHANG_ARCH_X64)
template <size_t Width>
static void interleave_matrix_rows(__m128i* rows) {
    __m128i next[16];
    for(size_t group = 0; group < 16; group += 2 * Width) {
        for(size_t i = 0; i < Width; i++) {
            const __m128i a = rows[group + i];
            const __m128i b = rows[group + Width + i];
            const size_t index = group + 2 * i;
            if constexpr (Width == 1) {
                next[index] = _mm_unpacklo_epi8(a, b);
                next[index + 1] = _mm_unpackhi_epi8(a, b);
            } else if constexpr (Width == 2) {
                next[index] = _mm_unpacklo_epi16(a, b);
                next[index + 1] = _mm_unpackhi_epi16(a, b);
            } else if constexpr (Width == 4) {
                next[index] = _mm_unpacklo_epi32(a, b);
                next[index + 1] = _mm_unpackhi_epi32(a, b);
            } else {
                next[index] = _mm_unpacklo_epi64(a, b);
                next[index + 1] = _mm_unpackhi_epi64(a, b);
            }
        }
    }
    for(size_t i = 0; i < 16; i++) rows[i] = next[i];
}
#endif

static void transpose_matrix_tile(const Block* input, size_t tile_rows, Block* output) {
#if defined(TAIHANG_ARCH_X64)
    const size_t column_blocks = tile_rows / 128;
    auto* bytes = reinterpret_cast<uint8_t*>(output);
    for(size_t row = 0; row < 128; row += 16) {
        for(size_t block = 0; block < column_blocks; block++) {
            __m128i rows[16];
            for(size_t i = 0; i < 16; i++) {
                rows[i] = input[(row + i) * column_blocks + block].mm;
            }
            interleave_matrix_rows<1>(rows);
            interleave_matrix_rows<2>(rows);
            interleave_matrix_rows<4>(rows);
            interleave_matrix_rows<8>(rows);
            for(size_t column = 0; column < 16; column++) {
                __m128i bits = rows[column];
                for(int bit = 7; bit >= 0; bit--) {
                    const uint16_t mask = static_cast<uint16_t>(_mm_movemask_epi8(bits));
                    const size_t output_row = block * 128 + column * 8 + bit;
                    std::memcpy(bytes + output_row * sizeof(Block) + row / 8,
                                &mask, sizeof(mask));
                    bits = _mm_slli_epi64(bits, 1);
                }
            }
        }
    }
#else
    bit_matrix_transpose(reinterpret_cast<const uint8_t*>(input), 128, tile_rows,
                         reinterpret_cast<uint8_t*>(output));
#endif
}

static void pack_selection_bits(const uint8_t* byte_data, size_t byte_len,
                         Block* block_data, size_t block_len) {
#if defined(TAIHANG_ARCH_X64)
    TAIHANG_ASSERT(byte_len == block_len * 128, "pack bits to blocks: size mismatch.");
    auto* out = reinterpret_cast<uint8_t*>(block_data);
    for(size_t i = 0; i < byte_len; i += 16) {
        const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_data + i));
        const unsigned mask = static_cast<unsigned>(~_mm_movemask_epi8(
            _mm_cmpeq_epi8(values, _mm_setzero_si128()))) & 0xffff;
        out[i / 8] = static_cast<uint8_t>(mask);
        out[i / 8 + 1] = static_cast<uint8_t>(mask >> 8);
    }
#else
    pack_bits_to_blocks(byte_data, byte_len, block_data, block_len);
#endif
}

// --- PublicParameters ---

std::string PublicParameters::format() const {
    std::ostringstream oss;
    oss << "[ALSZ OTE PublicParameters] \n"
        << "Malicious: " << (malicious ? "true" : "false") << "\n"
        << "Base Length: " << base_len << "\n"
        << base_ot_pp.format();
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const PublicParameters& pp) {
    os << pp.base_ot_pp << " " << pp.malicious << " " << pp.base_len;
    return os;
}

std::istream& operator>>(std::istream& is, PublicParameters& pp) {
    is >> pp.base_ot_pp >> pp.malicious >> pp.base_len;
    return is;
}

// --- Algorithms ---

PublicParameters setup(int curve_id, size_t base_len) {
    PublicParameters pp; 
    pp.malicious = false; 
    pp.base_ot_pp = taihang::mpc::np_ot::setup(curve_id); // Assuming standard integration
    pp.base_len = base_len;
    return pp;
}

static std::pair<std::vector<Block>, std::vector<Block>> random_sender_impl(
        net::NetIO& io, const PublicParameters& pp, size_t extend_len,
        const Block* vec_m0, const Block* vec_m1)
{
    /* 
    ** Phase 1: sender obtains a random blended matrix Q of matrix T and U from receiver
    ** T and U are tall, skinny matrices. To use base OT on their columns,
    ** the sender first obliviously obtains one of two keys per column from the receiver.
    ** The receiver then sends encryptions of the original and shared columns under k0 and k1, respectively.
    */

    // prepare to receive a secret shared matrix Q from receiver
    size_t row_num = extend_len;      // set row num as the length of ot extension
    size_t column_num = pp.base_len;  // set column num as the number of base ot

    TAIHANG_ASSERT(row_num % 128 == 0 && column_num % 128 == 0, "Row or column parameters are wrong");

    // initial OT phase (base OTs)
    // generate phase 1 selection bit vector (sparse form)
    prg::Seed seed = prg::set_seed(nullptr, 0); 
    std::vector<uint8_t> vec_sender_selection_bit = prg::gen_random_bits(seed, column_num); 

    // First receive one-out-of-two keys (actually seeds) from the receiver.
    auto vec_q_seed = taihang::mpc::np_ot::receiver(io, pp.base_ot_pp, vec_sender_selection_bit, column_num);

    TAIHANG_LOG("ALSZ OTE [step 1]:", std::format("Sender obliviously gets {} keys from Receiver via base OT", pp.base_len));

    // crush selection bit vector in sparse form to dense form
    std::vector<Block> vec_sender_selection_block(column_num / 128);
    pack_selection_bits(vec_sender_selection_bit.data(), column_num, vec_sender_selection_block.data(), column_num / 128);
    const auto& fixed_key = aes::get_fixed_key();

    if(column_num == 128) {
        const size_t matrix_batch_size = 2048;
        const size_t matrix_window_size = 32768;
        const size_t hash_batch_size = 256;
        std::vector<Block> vec_k0(row_num), vec_k1(row_num);
        std::vector<Block> matrix_p(std::min(matrix_window_size, row_num));
        prg::Seed column_seeds[128];
        for(size_t j = 0; j < column_num; j++) {
            prg::reset_seed(column_seeds[j], &vec_q_seed[j], 0);
        }
        std::exception_ptr io_error;

        #pragma omp parallel num_threads(config::thread_num)
        {
            Block matrix_q[matrix_batch_size];
            for(size_t window = 0; window < row_num; window += matrix_window_size) {
                const size_t window_rows = std::min(matrix_window_size, row_num - window);
                #pragma omp master
                {
                    try {
                        io.recv(matrix_p.data(), window_rows * sizeof(Block));
                    } catch(...) {
                        io_error = std::current_exception();
                    }
                }
                #pragma omp barrier
                if(io_error) break;
                #pragma omp for schedule(static)
                for(size_t tile = 0; tile < window_rows; tile += matrix_batch_size) {
                    const size_t offset = window + tile;
                    const size_t tile_rows = std::min(matrix_batch_size, window_rows - tile);
                    const size_t tile_blocks = tile_rows / 128;
                    for(size_t j = 0; j < column_num; j++) {
                        prg::Seed tile_seed = column_seeds[j];
                        tile_seed.counter = offset / 128;
                        Block* q_column = matrix_q + j * tile_blocks;
                        prg::gen_random_blocks(tile_seed, q_column, tile_blocks);
                        if(vec_sender_selection_bit[j] == 1) {
                            const Block* p_column = matrix_p.data() + tile + j * tile_blocks;
                            for(size_t i = 0; i < tile_blocks; i++) {
                                q_column[i] ^= p_column[i];
                            }
                        }
                    }

                    Block* k0 = vec_k0.data() + offset;
                    Block* k1 = vec_k1.data() + offset;
                    transpose_matrix_tile(matrix_q, tile_rows, k0);
                    for(size_t j = 0; j < tile_rows; j += hash_batch_size) {
                        const size_t batch_size = std::min(hash_batch_size, tile_rows - j);
                        for(size_t i = 0; i < batch_size; i++) {
                            k1[j + i] = k0[j + i] ^ vec_sender_selection_block[0];
                        }
                        aes::encrypt_ecb(fixed_key, k0 + j, k0 + j, batch_size);
                        aes::encrypt_ecb(fixed_key, k1 + j, k1 + j, batch_size);
                        if(vec_m0 != nullptr) {
                            for(size_t i = 0; i < batch_size; i++) {
                                k0[j + i] ^= vec_m0[offset + j + i];
                            }
                        }
                        if(vec_m1 != nullptr) {
                            for(size_t i = 0; i < batch_size; i++) {
                                k1[j + i] ^= vec_m1[offset + j + i];
                            }
                        }
                    }
                }
            }
        }
        if(io_error) std::rethrow_exception(io_error);
        return {std::move(vec_k0), std::move(vec_k1)};
    }

    const size_t column_blocks = row_num / 128;
    std::vector<Block> matrix_q(column_blocks * column_num); // size = ROW_NUM/128 * COLUMN_NUM
    for(size_t j = 0; j < column_num; j++) {
        prg::reset_seed(seed, &vec_q_seed[j], 0);
        Block* q_column = matrix_q.data() + j * column_blocks;
        prg::gen_random_blocks(seed, q_column, column_blocks);
    }

    // OT extension phase: receive u^i from receiver for every 1 \leq i \leq \kappa
    std::vector<Block> matrix_p((row_num / 128) * column_num);
    io.recv(matrix_p);

    // for every 1 \leq j \leq \kappa, define q^j = (s_j \cdot u^j) \oplus G(k_j^s_j)
    for(size_t j = 0; j < column_num; j++) {
        for(size_t i = 0; i < row_num / 128; i++) {
            if(vec_sender_selection_bit[j] == 1) {
                matrix_q[j * (row_num / 128) + i] ^= matrix_p[j * (row_num / 128) + i];
            }
        }
    }

    // prepare k_0 and k_1 for every row 1 \leq j \leq m
    const size_t row_blocks = column_num / 128;

    // transpose matrix Q
    std::vector<Block> matrix_q_transpose = std::move(matrix_p);
    bit_matrix_transpose(reinterpret_cast<uint8_t*>(matrix_q.data()), column_num, row_num, reinterpret_cast<uint8_t*>(matrix_q_transpose.data()));

    // Generate return vectors
    std::vector<Block> vec_k0 = std::move(matrix_q);
    vec_k0.resize(row_num);

    std::vector<Block> vec_k1(row_num);
    const size_t hash_batch_size = 8;
    #pragma omp parallel for num_threads(config::thread_num)
    for(size_t j = 0; j < row_num; j += hash_batch_size) {
        const size_t batch_size = std::min(hash_batch_size, row_num - j);
        Block states[2 * hash_batch_size];

        for(size_t b = 0; b < row_blocks; b++) {
            for(size_t i = 0; i < batch_size; i++) {
                const Block& q = matrix_q_transpose[(j + i) * row_blocks + b];
                states[2 * i] ^= q;
                states[2 * i + 1] ^= q ^ vec_sender_selection_block[b];
            }
            aes::encrypt_ecb(fixed_key, states, states, 2 * batch_size);
        }

        if(vec_m0 != nullptr) {
            for(size_t i = 0; i < batch_size; i++) {
                states[2 * i] ^= vec_m0[j + i];
            }
        }
        if(vec_m1 != nullptr) {
            for(size_t i = 0; i < batch_size; i++) {
                states[2 * i + 1] ^= vec_m1[j + i];
            }
        }
        for(size_t i = 0; i < batch_size; i++) {
            vec_k0[j + i] = states[2 * i];
            vec_k1[j + i] = states[2 * i + 1];
        }
    }

    return {std::move(vec_k0), std::move(vec_k1)};
}

// sender obtains extend_len number of key pairs
std::pair<std::vector<Block>, std::vector<Block>> random_sender(
        net::NetIO& io, const PublicParameters& pp, size_t extend_len)
{
    return random_sender_impl(io, pp, extend_len, nullptr, nullptr);
}

// implement random receive: note this random ot is slightly different from Beaver's ROT
// cause receiver can choose selection bit itself
// receiver only obtains extend_len number of 1-out-of-2 keys
std::vector<Block> random_receiver(net::NetIO& io, const PublicParameters& pp, const std::vector<uint8_t>& vec_receiver_selection_bit, size_t extend_len) 
{
    // prepare random sharing of repetition of r 
    size_t row_num = extend_len; 
    size_t column_num = pp.base_len; 
    TAIHANG_ASSERT(row_num % 128 == 0 && column_num % 128 == 0, "Row or column parameters are wrong");

    prg::Seed seed = prg::set_seed(nullptr, 0); 

    std::vector<Block> vec_t_seed = prg::gen_random_blocks(seed, column_num);
    std::vector<Block> vec_u_seed = prg::gen_random_blocks(seed, column_num);

    taihang::mpc::np_ot::sender(io, pp.base_ot_pp, vec_t_seed, vec_u_seed, column_num); 

    TAIHANG_LOG("ALSZ OTE [step 1]:", std::format("Receiver transmits {} seeds to Sender via base OT", column_num));
    
    const auto& fixed_key = aes::get_fixed_key();
    if(column_num == 128) {
        const size_t matrix_batch_size = 2048;
        const size_t matrix_window_size = 32768;
        std::vector<Block> vec_k(row_num);
        std::vector<Block> matrix_p(std::min(matrix_window_size, row_num));
        prg::Seed t_seeds[128], u_seeds[128];
        for(size_t j = 0; j < column_num; j++) {
            prg::reset_seed(t_seeds[j], &vec_t_seed[j], 0);
            prg::reset_seed(u_seeds[j], &vec_u_seed[j], 0);
        }
        io.flush();
        std::exception_ptr io_error;

        #pragma omp parallel num_threads(config::thread_num)
        {
            Block matrix_t[matrix_batch_size];
            Block selection[matrix_batch_size / 128];
            for(size_t window = 0; window < row_num; window += matrix_window_size) {
                const size_t window_rows = std::min(matrix_window_size, row_num - window);
                #pragma omp for schedule(static)
                for(size_t tile = 0; tile < window_rows; tile += matrix_batch_size) {
                    const size_t offset = window + tile;
                    const size_t tile_rows = std::min(matrix_batch_size, window_rows - tile);
                    const size_t tile_blocks = tile_rows / 128;
                    pack_selection_bits(vec_receiver_selection_bit.data() + offset, tile_rows,
                                        selection, tile_blocks);
                    for(size_t j = 0; j < column_num; j++) {
                        prg::Seed t_seed = t_seeds[j];
                        prg::Seed u_seed = u_seeds[j];
                        t_seed.counter = u_seed.counter = offset / 128;
                        Block* t_column = matrix_t + j * tile_blocks;
                        Block* p_column = matrix_p.data() + tile + j * tile_blocks;
                        prg::gen_random_blocks(t_seed, t_column, tile_blocks);
                        prg::gen_random_blocks(u_seed, p_column, tile_blocks);
                        for(size_t i = 0; i < tile_blocks; i++) {
                            p_column[i] ^= t_column[i] ^ selection[i]; // p = t xor u xor r
                        }
                    }
                    Block* output = vec_k.data() + offset;
                    transpose_matrix_tile(matrix_t, tile_rows, output);
                    aes::encrypt_ecb(fixed_key, output, output, tile_rows);
                }
                #pragma omp master
                {
                    try {
                        io.send(matrix_p.data(), window_rows * sizeof(Block));
                    } catch(...) {
                        io_error = std::current_exception();
                    }
                }
                #pragma omp barrier
                if(io_error) break;
            }
        }
        if(io_error) std::rethrow_exception(io_error);
        TAIHANG_LOG("ALSZ OTE [step 2]:", std::format("Receiver ===> {}*{} adjust bit matrix ===> Sender [{:.2f} MB]",
                                                      row_num, column_num,
                                                      static_cast<double>(row_num / 128 * column_num * 16) / (1024 * 1024)));
        return vec_k;
    }

    // crush sparse bytes to dense bytes
    std::vector<Block> vec_receiver_selection_block(row_num / 128);
    pack_selection_bits(vec_receiver_selection_bit.data(), row_num, vec_receiver_selection_block.data(), row_num / 128);

    // derive matrix T, U, and P: size = ROW_NUM/128*COLUMN_NUM
    const size_t column_blocks = row_num / 128;
    std::vector<Block> matrix_t(column_blocks * column_num);
    std::vector<Block> matrix_p(column_blocks * column_num);
    for(size_t j = 0; j < column_num; j++) {
        // Generate two random matrices from seeds.
        Block* t_column = matrix_t.data() + j * column_blocks;
        Block* p_column = matrix_p.data() + j * column_blocks;
        prg::reset_seed(seed, &vec_t_seed[j], 0);
        prg::gen_random_blocks(seed, t_column, column_blocks); // t = G(t_seed)
        prg::reset_seed(seed, &vec_u_seed[j], 0);
        prg::gen_random_blocks(seed, p_column, column_blocks); // u = G(u_seed)
        for(size_t i = 0; i < column_blocks; i++) {
            p_column[i] ^= t_column[i] ^ vec_receiver_selection_block[i]; // p = t xor u xor r
        }
    }

    // transmit adjust bit matrix 
    if (!matrix_p.empty()) {
        io.flush();
        io.send(matrix_p.data(), matrix_p.size() * sizeof(Block));
    }
    
    TAIHANG_LOG("ALSZ OTE [step 2]:", std::format("Receiver ===> {}*{} adjust bit matrix ===> Sender [{:.2f} MB]", 
                                                  row_num, column_num, 
                                                  static_cast<double>(row_num / 128 * column_num * 16) / (1024 * 1024)));

    // transpose matrix T                                              
    std::vector<Block> matrix_t_transpose = std::move(matrix_p);

    const size_t row_blocks = column_num / 128;
    bit_matrix_transpose(reinterpret_cast<uint8_t*>(matrix_t.data()), column_num, row_num, reinterpret_cast<uint8_t*>(matrix_t_transpose.data()));

    std::vector<Block> vec_k = std::move(matrix_t);
    vec_k.resize(row_num);
    const size_t hash_batch_size = 8;
    #pragma omp parallel for num_threads(config::thread_num)
    for(size_t i = 0; i < row_num; i += hash_batch_size) {
        const size_t batch_size = std::min(hash_batch_size, row_num - i);
        Block states[hash_batch_size];

        for(size_t b = 0; b < row_blocks; b++) {
            for(size_t j = 0; j < batch_size; j++) {
                states[j] ^= matrix_t_transpose[(i + j) * row_blocks + b];
            }
            aes::encrypt_ecb(fixed_key, states, states, batch_size);
        }

        for(size_t j = 0; j < batch_size; j++) {
            vec_k[i + j] = states[j];
        }
    } 
    return vec_k; 
}

template <typename Policy>
void sender(net::NetIO& io, const PublicParameters& pp, 
            const std::vector<typename Policy::Message>& vec_m0, 
            const std::vector<typename Policy::Message>& vec_m1, 
            size_t extend_len) 
{
    TAIHANG_TIMER("ALSZ OTE:", "Sender total time");

    std::vector<typename Policy::Ciphertext> vec_outer_c0, vec_outer_c1;

    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        std::tie(vec_outer_c0, vec_outer_c1) = random_sender_impl(
            io, pp, extend_len, vec_m0.data(), vec_m1.data());
    } else {
        // obtain two vector<Block> of length extend_len
        std::vector<Block> vec_k0, vec_k1;
        std::tie(vec_k0, vec_k1) = random_sender(io, pp, extend_len);
        vec_outer_c0.resize(extend_len);
        vec_outer_c1.resize(extend_len);
        #pragma omp parallel for num_threads(config::thread_num)
        for(size_t i = 0; i < extend_len; i++) {
            vec_outer_c0[i] = Policy::encrypt(vec_k0[i], vec_m0[i]);
            vec_outer_c1[i] = Policy::encrypt(vec_k1[i], vec_m1[i]);
        }
    }

    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        if (!vec_outer_c0.empty()) {
            io.flush();
            io.send(vec_outer_c0.data(), vec_outer_c0.size() * sizeof(Block));
            io.send(vec_outer_c1.data(), vec_outer_c1.size() * sizeof(Block));
        }
    } else {
        io.send(vec_outer_c0);
        io.send(vec_outer_c1);
    }

    [[maybe_unused]] size_t item_size = CiphertextTraits<typename Policy::Ciphertext>::get_size(vec_outer_c0[0]);
    
    TAIHANG_LOG("ALSZ OTE [step 3]:", std::format("Sender ===> (vec_c0, vec_c1) ===> Receiver [{:.2f} MB]", 
                                                  static_cast<double>(extend_len * item_size * 2) / (1024 * 1024))); 
}

template <typename Policy>
std::vector<typename Policy::Message> receiver(net::NetIO& io, const PublicParameters& pp, 
                                               const std::vector<uint8_t>& vec_receiver_selection_bit, 
                                               size_t extend_len) 
{
    TAIHANG_TIMER("ALSZ OTE:", "Receiver total time");

    // receive extend_len number of key
    std::vector<Block> vec_k = random_receiver(io, pp, vec_receiver_selection_bit, extend_len); 

    std::vector<typename Policy::Message> vec_result;

    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        const size_t ciphertext_batch_size = 16384;
        vec_result = std::move(vec_k);
        std::vector<Block> vec_outer_c(std::min(ciphertext_batch_size, extend_len));

        for(size_t choice = 0; choice < 2; choice++) {
            for(size_t offset = 0; offset < extend_len; offset += ciphertext_batch_size) {
                const size_t batch_size = std::min(ciphertext_batch_size, extend_len - offset);
                io.recv(vec_outer_c.data(), batch_size * sizeof(Block));
                #pragma omp parallel for num_threads(config::thread_num)
                for(size_t i = 0; i < batch_size; i++) {
                    const size_t index = offset + i;
                    const bool selected = static_cast<size_t>(vec_receiver_selection_bit[index] != 0) == choice;
#if defined(TAIHANG_ARCH_X64)
                    const __m128i mask = _mm_set1_epi32(-static_cast<int>(selected));
                    vec_result[index].mm = _mm_xor_si128(vec_result[index].mm,
                        _mm_and_si128(vec_outer_c[i].mm, mask));
#else
                    const uint8x16_t mask = vdupq_n_u8(static_cast<uint8_t>(-static_cast<int>(selected)));
                    vec_result[index].mm = veorq_u8(vec_result[index].mm,
                        vandq_u8(vec_outer_c[i].mm, mask));
#endif
                }
            }
        }
    } else {
        std::vector<typename Policy::Ciphertext> vec_outer_c0(extend_len), vec_outer_c1(extend_len);
        io.recv(vec_outer_c0);
        io.recv(vec_outer_c1);
        vec_result.resize(extend_len);

        #pragma omp parallel for num_threads(config::thread_num)
        for(size_t i = 0; i < extend_len; i++) {
            if(vec_receiver_selection_bit[i] == 0) {
                vec_result[i] = Policy::decrypt(vec_k[i], vec_outer_c0[i]);
            } else {
                vec_result[i] = Policy::decrypt(vec_k[i], vec_outer_c1[i]);
            }
        }
    }   

    TAIHANG_LOG("ALSZ OTE [step 4]:", "Receiver obtains vec_m"); 
    // return std::move(vec_result);
    return vec_result;
}

// one message is dummy
template <typename Policy>
void onesided_sender(net::NetIO& io, const PublicParameters& pp, const std::vector<typename Policy::Message>& vec_m, size_t extend_len) 
{
    TAIHANG_TIMER("ALSZ OTE:", "Sender total time");

    std::vector<typename Policy::Ciphertext> vec_outer_c;

    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        std::vector<Block> vec_k0, vec_k1;
        std::tie(vec_k0, vec_k1) = random_sender_impl(
            io, pp, extend_len, nullptr, vec_m.data());
        vec_outer_c = std::move(vec_k1);
    } else {
        // obtain extend_len number of two keys
        std::vector<Block> vec_k0, vec_k1;
        std::tie(vec_k0, vec_k1) = random_sender(io, pp, extend_len);
        vec_outer_c.resize(extend_len);
        #pragma omp parallel for num_threads(config::thread_num) shared(vec_k1)
        for(size_t i = 0; i < extend_len; i++) {
            vec_outer_c[i] = Policy::encrypt(vec_k1[i], vec_m[i]);
        }
    }
    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        if (!vec_outer_c.empty()) {
            io.flush();
            io.send(vec_outer_c.data(), vec_outer_c.size() * sizeof(Block));
        }
    } else {
        io.send(vec_outer_c);
    }

    [[maybe_unused]] size_t item_size = CiphertextTraits<typename Policy::Ciphertext>::get_size(vec_outer_c[0]);

    TAIHANG_LOG("ALSZ OTE [step 3]:", std::format("Sender ===> vec_c ===> Receiver [{:.2f} MB]", 
                                                  static_cast<double>(extend_len * item_size) / (1024 * 1024)));
}

template <typename Policy>
std::vector<typename Policy::Message> onesided_receiver(net::NetIO& io, const PublicParameters& pp, 
                                                        const std::vector<uint8_t>& vec_receiver_selection_bit, 
                                                        size_t extend_len) 
{
    TAIHANG_TIMER("ALSZ OTE:", "Receiver total time");
 
    // obtain extend_len number of keys
    std::vector<Block> vec_k = random_receiver(io, pp, vec_receiver_selection_bit, extend_len);

    std::vector<typename Policy::Message> vec_result;

    if constexpr (std::is_same_v<Policy, BlockPolicy>) {
        const size_t ciphertext_batch_size = 16384;
        vec_result = std::move(vec_k);
        std::vector<Block> vec_outer_c(std::min(ciphertext_batch_size, extend_len));
        size_t selected_num = 0;
        for(size_t offset = 0; offset < extend_len; offset += ciphertext_batch_size) {
            const size_t batch_size = std::min(ciphertext_batch_size, extend_len - offset);
            io.recv(vec_outer_c.data(), batch_size * sizeof(Block));
            for(size_t i = 0; i < batch_size; i++) {
                const size_t index = offset + i;
                if(vec_receiver_selection_bit[index] == 1) {
                    vec_result[selected_num++] = Policy::decrypt(vec_result[index], vec_outer_c[i]);
                }
            }
        }
        vec_result.resize(selected_num);
    } else {
        std::vector<typename Policy::Ciphertext> vec_outer_c(extend_len);
        io.recv(vec_outer_c);
        vec_result.reserve(extend_len);
        for(size_t i = 0; i < extend_len; i++) {
            if(vec_receiver_selection_bit[i] == 1) {
                vec_result.emplace_back(Policy::decrypt(vec_k[i], vec_outer_c[i]));
            }
        }
    }   

    TAIHANG_LOG("ALSZ OTE [step 4]:", "Receiver obtains vec_m"); 
    // return std::move(vec_result);
    return vec_result;
}

}

// Add this to the bottom of alsz_ote.cpp

namespace taihang::mpc::alsz_ote {

// Explicit instantiations for BlockPolicy
template void sender<BlockPolicy>(
    net::NetIO& io, 
    const PublicParameters& pp, 
    const std::vector<typename BlockPolicy::Message>& vec_m0, 
    const std::vector<typename BlockPolicy::Message>& vec_m1, 
    size_t extend_len);

template std::vector<typename BlockPolicy::Message> receiver<BlockPolicy>(
    net::NetIO& io, 
    const PublicParameters& pp, 
    const std::vector<uint8_t>& vec_receiver_selection_bit, 
    size_t extend_len);

// Explicit instantiations for BytesPolicy (To prevent similar issues later)
template void sender<BytesPolicy>(
    net::NetIO& io, 
    const PublicParameters& pp, 
    const std::vector<typename BytesPolicy::Message>& vec_m0, 
    const std::vector<typename BytesPolicy::Message>& vec_m1, 
    size_t extend_len);

template std::vector<typename BytesPolicy::Message> receiver<BytesPolicy>(
    net::NetIO& io, 
    const PublicParameters& pp, 
    const std::vector<uint8_t>& vec_receiver_selection_bit, 
    size_t extend_len);


// at the bottom of alsz_ote.cpp
template void alsz_ote::onesided_sender<alsz_ote::BlockPolicy>(
    net::NetIO&, const PublicParameters&,
    const std::vector<BlockPolicy::Message>&, size_t);

template std::vector<BlockPolicy::Message>
alsz_ote::onesided_receiver<alsz_ote::BlockPolicy>(
    net::NetIO&, const PublicParameters&,
    const std::vector<uint8_t>&, size_t);

} // namespace taihang::mpc::alsz_ote
