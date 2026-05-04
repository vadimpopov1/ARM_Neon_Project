#include <arm_neon.h>
#include <cstdint>
#include <iostream>
#include <chrono>

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

int64_t process_array_neon(const int32_t* __restrict__ data, size_t n) {
    int64_t sum = 0;
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);

        int32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        int32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));

        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);

        int32x4_t pos_part = vandq_s32(vec, mask_pos);
        int32x4_t neg_part = vandq_s32(abs_val, mask_neg);
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);

        acc = vaddq_s32(acc, contrib);
    }

    sum = vaddlvq_s32(acc);

    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }

    return sum;
}

int main() {
    const size_t N = 1000000;
    alignas(16) static int32_t data[N];
    for (size_t i = 0; i < N; ++i)
        data[i] = (int32_t)(i % 7) - 3;

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t r1 = process_array_scalar(data, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    int64_t r2 = process_array_neon(data, N);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto scalar_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto neon_us   = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Скалярная реализация: " << r1 << " (" << scalar_us / 1e6 << " s)\n";
    std::cout << "Реализация NEON: " << r2 << " (" << neon_us   / 1e6 << " s)\n";
    std::cout << "Ускорение в: " << (double)scalar_us / neon_us << " раз.\n";

    return (r1 == r2) ? 0 : 1;
}