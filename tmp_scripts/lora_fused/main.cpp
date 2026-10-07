// Standalone driver for add_lora_fused (z1 shrink + z2 expand). No torch.
//   ./bench <B> <H1> <R> <nSlices> <iters> <slots> <ngroups>
// Slices are equal-width: h2[s] = H1 / nSlices, so yWidth = H1.
#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <time.h>

#include "types.h"

namespace vllm_ascend {
extern void add_lora_fused_impl(AscendType type, void *stream, void *x, void *const *wa,
                                void *const *wb, void *indices, void *y, void *z1ws,
                                uint32_t batch, uint32_t H1, uint32_t R, const uint32_t *h2,
                                uint32_t nSlices, uint32_t yWidth, float scale,
                                uint32_t addInputs, uint32_t aivNum);
}

#define ACL_CHECK(expr)                                                                      \
    do {                                                                                     \
        aclError _e = (expr);                                                                 \
        if (_e != ACL_SUCCESS) {                                                              \
            fprintf(stderr, "%s:%d ACL error %d on %s\n", __FILE__, __LINE__, _e, #expr);      \
            exit(1);                                                                          \
        }                                                                                    \
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
    uint32_t B = (argc > 1) ? (uint32_t)atoi(argv[1]) : 48;
    uint32_t H1 = (argc > 2) ? (uint32_t)atoi(argv[2]) : 4096;
    uint32_t R = (argc > 3) ? (uint32_t)atoi(argv[3]) : 16;
    uint32_t S = (argc > 4) ? (uint32_t)atoi(argv[4]) : 1;
    int iters = (argc > 5) ? atoi(argv[5]) : 200;
    const uint32_t SLOTS = (argc > 6) ? (uint32_t)atoi(argv[6]) : 4;
    const float scale = 0.5f;

    uint32_t h2[4] = {0, 0, 0, 0};
    for (uint32_t s = 0; s < S; s++) h2[s] = H1 / S;
    uint32_t yWidth = 0;
    for (uint32_t s = 0; s < S; s++) yWidth += h2[s];

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    int64_t aiv = 0;
    if (aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv) != ACL_SUCCESS || aiv <= 0)
        aiv = 40;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-1.f, 1.f);

    std::vector<uint16_t> hx((size_t)B * H1);
    for (auto &v : hx) v = f32_to_bf16(d(rng));
    std::vector<std::vector<uint16_t>> hwa(S), hwb(S);
    for (uint32_t s = 0; s < S; s++) {
        hwa[s].resize((size_t)SLOTS * R * H1);
        hwb[s].resize((size_t)SLOTS * h2[s] * R);
        for (auto &v : hwa[s]) v = f32_to_bf16(d(rng) * 0.05f);
        for (auto &v : hwb[s]) v = f32_to_bf16(d(rng) * 0.05f);
    }
    // argv[7] = NGROUPS: 0 (default) keeps the historical round-robin index
    // pattern (every row a different slot -- worst case for weight reuse);
    // NGROUPS > 0 models PRODUCTION, where MoeInitRoutingV3 has already sorted
    // the rows by expert, so the slot index is monotone with runs of B/NGROUPS
    // consecutive rows sharing one slot. The MoE w13 decode point is
    // B=288 SLOTS=96 NGROUPS=32 (32 local experts, one active adapter).
    const uint32_t NGROUPS = (argc > 7) ? (uint32_t)atoi(argv[7]) : 0;
    std::vector<int64_t> hidx(B);
    if (NGROUPS == 0) {
        for (uint32_t t = 0; t < B; t++) hidx[t] = (t % 7 == 6) ? -1 : (int64_t)(t % SLOTS);
    } else {
        for (uint32_t t = 0; t < B; t++)
            hidx[t] = (int64_t)(((uint64_t)t * NGROUPS / B) % SLOTS);
    }
    std::vector<uint16_t> hy((size_t)B * yWidth, f32_to_bf16(0.f));

    void *dx, *didx, *dy, *dz1, *dwa[4], *dwb[4];
    ACL_CHECK(aclrtMalloc(&dx, hx.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&didx, hidx.size() * 8, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dy, hy.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dz1, (size_t)S * B * R * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dx, hx.size() * 2, hx.data(), hx.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(didx, hidx.size() * 8, hidx.data(), hidx.size() * 8,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t s = 0; s < S; s++) {
        ACL_CHECK(aclrtMalloc(&dwa[s], hwa[s].size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&dwb[s], hwb[s].size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(dwa[s], hwa[s].size() * 2, hwa[s].data(), hwa[s].size() * 2,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(dwb[s], hwb[s].size() * 2, hwb[s].data(), hwb[s].size() * 2,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }
    for (uint32_t s = S; s < 4; s++) { dwa[s] = dwa[0]; dwb[s] = dwb[0]; }

    auto launch = [&]() {
        ACL_CHECK(aclrtMemcpy(dy, hy.size() * 2, hy.data(), hy.size() * 2,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        vllm_ascend::add_lora_fused_impl(vllm_ascend::AscendType::BF16, stream, dx, dwa, dwb, didx,
                                        dy, dz1, B, H1, R, h2, S, yWidth, scale, 1,
                                        (uint32_t)aiv);
    };
    auto launch_timed = [&]() {
        vllm_ascend::add_lora_fused_impl(vllm_ascend::AscendType::BF16, stream, dx, dwa, dwb, didx,
                                        dy, dz1, B, H1, R, h2, S, yWidth, scale, 1,
                                        (uint32_t)aiv);
    };

    // optional HBM load generator: large device-to-device copies on a second
    // stream. Uses the DMA engine, not the vector cores, so it contends for
    // memory the way the base grouped matmul does.
    const char *memload = getenv("LF_MEMLOAD_MB");
    if (memload && *memload == 0) memload = nullptr;
    aclrtStream loadStream = nullptr;
    void *lsrc = nullptr, *ldst = nullptr;
    size_t lbytes = 0;
    if (memload) {
        lbytes = (size_t)atoi(memload) * 1024ull * 1024ull;
        ACL_CHECK(aclrtCreateStream(&loadStream));
        ACL_CHECK(aclrtMalloc(&lsrc, lbytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&ldst, lbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    auto pump = [&](int n) {
        if (!memload) return;
        for (int i = 0; i < n; i++)
            aclrtMemcpyAsync(ldst, lbytes, lsrc, lbytes, ACL_MEMCPY_DEVICE_TO_DEVICE, loadStream);
    };

    launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));

    const bool do_check = getenv("LF_NOCHECK") == nullptr;
    // ---- z1 check in isolation: read the GM workspace [S][B][R] ----
    if (do_check) {
        std::vector<float> gz1((size_t)S * B * R);
        ACL_CHECK(aclrtMemcpy(gz1.data(), gz1.size() * 4, dz1, gz1.size() * 4,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        double z1max = 0.0, z1rel = 0.0;
        int bad = 0;
        for (uint32_t s = 0; s < S; s++) {
            for (uint32_t t = 0; t < (B < 16 ? B : 16); t++) {
                int64_t slot = hidx[t];
                if (slot < 0) continue;
                for (uint32_t r = 0; r < R; r++) {
                    double want = 0.0;
                    for (uint32_t hh = 0; hh < H1; hh++)
                        want += (double)bf16_to_f32(hx[(size_t)t * H1 + hh])
                                * bf16_to_f32(hwa[s][((size_t)slot * R + r) * H1 + hh]);
                    want *= scale;
                    double got = gz1[((size_t)s * B + t) * R + r];
                    double a = fabs(got - want);
                    if (a > z1max) z1max = a;
                    double den = fabs(want) > 1e-3 ? fabs(want) : 1e-3;
                    if (a / den > z1rel) z1rel = a / den;
                    if (a / den > 0.05) bad++;
                }
            }
        }
        printf("  z1: max|err|=%.4g rel=%.4g bad=%d\n", z1max, z1rel, bad);
    }

    // ---- correctness vs fp32 host reference ----
    std::vector<uint16_t> gy(hy.size());
    double maxrel = 0.0, maxabs = 0.0;
    if (do_check) {
    ACL_CHECK(aclrtMemcpy(gy.data(), gy.size() * 2, dy, gy.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t checkB = B < 64 ? B : 64;
    // precompute z1[t][s][r] once; recomputing it per output column made this
    // reference O(h2*R*H1) per row and dominated the whole run
    std::vector<double> z1ref((size_t)checkB * S * R, 0.0);
    for (uint32_t t = 0; t < checkB; t++) {
        int64_t slot = hidx[t];
        if (slot < 0) continue;
        for (uint32_t s = 0; s < S; s++)
            for (uint32_t r = 0; r < R; r++) {
                double z = 0.0;
                for (uint32_t hh = 0; hh < H1; hh++)
                    z += (double)bf16_to_f32(hx[(size_t)t * H1 + hh])
                         * bf16_to_f32(hwa[s][((size_t)slot * R + r) * H1 + hh]);
                z1ref[((size_t)t * S + s) * R + r] = z * scale;
            }
    }
    for (uint32_t t = 0; t < checkB; t++) {
        int64_t slot = hidx[t];
        uint32_t off = 0;
        for (uint32_t s = 0; s < S; s++) {
            for (uint32_t j = 0; j < h2[s]; j++) {
                double want = 0.0;
                if (slot >= 0)
                    for (uint32_t r = 0; r < R; r++)
                        want += z1ref[((size_t)t * S + s) * R + r]
                                * bf16_to_f32(hwb[s][((size_t)slot * h2[s] + j) * R + r]);
                double got = bf16_to_f32(gy[(size_t)t * yWidth + off + j]);
                double a = fabs(got - want);
                if (a > maxabs) maxabs = a;
                double den = fabs(want) > 1e-3 ? fabs(want) : 1e-3;
                if (a / den > maxrel) maxrel = a / den;
            }
            off += h2[s];
        }
    }
    }
    printf("B=%u H1=%u R=%u S=%u slots=%u grp=%u | max|err|=%.4g rel=%.4g", B, H1, R, S, SLOTS, NGROUPS, maxabs, maxrel);

    pump(400);
    for (int i = 0; i < 10; i++) launch_timed();
    ACL_CHECK(aclrtSynchronizeStream(stream));
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) { pump(4); launch_timed(); }
    double ts = now_ms();
    ACL_CHECK(aclrtSynchronizeStream(stream));
    double t1 = now_ms();
    printf(" | %.2f us/call | submit %.2f us%s\n", (t1 - t0) * 1e3 / iters,
           (ts - t0) * 1e3 / iters,
           (ts - t0) > 0.9 * (t1 - t0) ? "  <<< HOST BOUND" : "");
    fflush(stdout);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
