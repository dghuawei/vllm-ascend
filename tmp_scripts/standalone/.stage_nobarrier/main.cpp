// Standalone driver for add_lora_swiglu_quant. No torch, no vllm.
// Usage: ./bench [T] [W] [has_delta] [iters]
//
// bf16 only: bf16 <-> fp32 is a pure bit shift, so the host reference needs no
// library support and the comparison is exact about what the kernel was fed.

#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "types.h"

namespace vllm_ascend {
extern void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up, void *delta,
                                       void *act, void *y, void *scale, uint32_t batch,
                                       uint32_t width, uint32_t has_delta, uint32_t aiv_num);
}

#define ACL_CHECK(expr)                                                                  \
    do {                                                                                 \
        aclError _err = (expr);                                                          \
        if (_err != ACL_SUCCESS) {                                                       \
            fprintf(stderr, "%s:%d ACL error %d on: %s\n", __FILE__, __LINE__, _err, #expr); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static inline uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t h)
{
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static double now_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    uint32_t T = (argc > 1) ? (uint32_t)atoi(argv[1]) : 8192;
    uint32_t W = (argc > 2) ? (uint32_t)atoi(argv[2]) : 2048;
    uint32_t has_delta = (argc > 3) ? (uint32_t)atoi(argv[3]) : 1;
    int iters = (argc > 4) ? atoi(argv[4]) : 50;
    float amp = (argc > 5) ? (float)atof(argv[5]) : 2.0f;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    int64_t aiv = 0;
    if (aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv) != ACL_SUCCESS || aiv <= 0) {
        aiv = 40;
    }
    uint32_t aiv_num = (uint32_t)aiv;

    const size_t n_gu = (size_t)T * 2 * W;
    const size_t n_out = (size_t)T * W;

    std::vector<uint16_t> h_gu(n_gu), h_delta(n_gu);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n_gu; i++) {
        h_gu[i] = f32_to_bf16(dist(rng));
        h_delta[i] = f32_to_bf16(dist(rng) * 0.1f);
    }

    void *d_gu = nullptr, *d_delta = nullptr, *d_act = nullptr, *d_y = nullptr, *d_scale = nullptr;
    ACL_CHECK(aclrtMalloc(&d_gu, n_gu * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_delta, n_gu * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_act, n_out * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_y, n_out, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_scale, (size_t)T * 4, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(d_gu, n_gu * 2, h_gu.data(), n_gu * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_delta, n_gu * 2, h_delta.data(), n_gu * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    auto launch = [&]() {
        vllm_ascend::add_lora_swiglu_quant_impl(
            vllm_ascend::AscendType::BF16, stream, d_gu, has_delta ? d_delta : nullptr, d_act, d_y,
            d_scale, T, W, has_delta, aiv_num);
    };

    launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ---- correctness against an fp32 host reference ----
    std::vector<uint16_t> h_act(n_out);
    std::vector<int8_t> h_y(n_out);
    std::vector<float> h_scale(T);
    ACL_CHECK(aclrtMemcpy(h_act.data(), n_out * 2, d_act, n_out * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(h_y.data(), n_out, d_y, n_out, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(h_scale.data(), (size_t)T * 4, d_scale, (size_t)T * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    // Two references. mode 0 accumulates the delta in fp32 (what the current
    // kernel does); mode 1 rounds the sum to bf16 first (what the eager torch
    // chain does, and what a bf16-add kernel variant would do). A variant is
    // judged against whichever it implements, so both are reported.
    uint32_t check_rows = T < 512 ? T : 512;
    double checked = (double)check_rows * W;
    std::vector<float> ref(W);
    for (int mode = 0; mode < 2; mode++) {
        double max_act_err = 0.0, max_scale_rel = 0.0;
        long y_off_by_1 = 0, y_off_more = 0;
        for (uint32_t t = 0; t < check_rows; t++) {
            float amax = 0.0f;
            for (uint32_t j = 0; j < W; j++) {
                float g = bf16_to_f32(h_gu[(size_t)t * 2 * W + j]);
                float u = bf16_to_f32(h_gu[(size_t)t * 2 * W + W + j]);
                if (has_delta) {
                    g += bf16_to_f32(h_delta[(size_t)t * 2 * W + j]);
                    u += bf16_to_f32(h_delta[(size_t)t * 2 * W + W + j]);
                    if (mode == 1) {
                        g = bf16_to_f32(f32_to_bf16(g));
                        u = bf16_to_f32(f32_to_bf16(u));
                    }
                }
                float a = u * (g / (1.0f + expf(-g)));
                ref[j] = a;
                float aa = fabsf(a);
                if (aa > amax) amax = aa;
            }
            float s = amax / 127.0f;
            double d = fabs((double)h_scale[t] - (double)s) / (s > 0 ? s : 1.0);
            if (d > max_scale_rel) max_scale_rel = d;
            for (uint32_t j = 0; j < W; j++) {
                float got = bf16_to_f32(h_act[(size_t)t * W + j]);
                double e = fabs(got - ref[j]);
                if (e > max_act_err) max_act_err = e;
                int want = (s > 0) ? (int)lrintf(ref[j] / s) : 0;
                if (want > 127) want = 127;
                if (want < -127) want = -127;
                int diff = abs((int)h_y[(size_t)t * W + j] - want);
                if (diff == 1) y_off_by_1++;
                else if (diff > 1) y_off_more++;
            }
        }
        printf("ref=%-8s max|act-ref|=%-10.4g scale_rel=%-10.3g y_off1=%-8.3g%% y_off>1=%.3g%%\n",
               mode == 0 ? "fp32add" : "bf16add", max_act_err, max_scale_rel,
               100.0 * y_off_by_1 / checked, 100.0 * y_off_more / checked);
    }
    { // non-finite guard: half-precision intermediates can overflow in exp()
        long bad = 0;
        for (size_t i = 0; i < n_out; i++) {
            if (!std::isfinite(bf16_to_f32(h_act[i]))) bad++;
        }
        for (uint32_t t = 0; t < T; t++) {
            if (!std::isfinite(h_scale[t])) bad++;
        }
        if (bad) printf("  !! %ld non-finite outputs\n", bad);
    }

    // ---- timing ----
    for (int i = 0; i < 10; i++) launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));
    double t1 = now_ms();

    double us = (t1 - t0) * 1000.0 / iters;
    double bytes = (double)T * W * (has_delta ? 11.0 : 7.0) + (double)T * 4.0;
    printf("T=%u W=%u delta=%u aiv=%u | %.1f us | %.1f GB/s\n", T, W, has_delta, aiv_num, us,
           bytes / (us * 1e-6) / 1e9);

    ACL_CHECK(aclrtFree(d_gu));
    ACL_CHECK(aclrtFree(d_delta));
    ACL_CHECK(aclrtFree(d_act));
    ACL_CHECK(aclrtFree(d_y));
    ACL_CHECK(aclrtFree(d_scale));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
