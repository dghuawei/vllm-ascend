#!/usr/bin/env python3
"""Generate the BASELINE harness (one interleaved [T,2W] delta) from the current
two-buffer main.cpp by reverting exactly the four two-delta edits.

Derived rather than taken from .stage_matrix/main.cpp, because that copy predates
the swiglu_limit argument and so cannot link against the HEAD kernel. Everything
except the delta plumbing is then identical between the two variants.

usage: make_main_base.py <main.cpp> <out main_base.cpp>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

EDITS = [
    # 1. impl declaration
    ("""extern void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up,
                                       void *delta_gate, void *delta_up, void *act, void *y,
                                       void *scale, uint32_t batch, uint32_t width,
                                       uint32_t has_delta, float swiglu_limit, uint32_t aiv_num);""",
     """extern void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up,
                                       void *delta, void *act, void *y,
                                       void *scale, uint32_t batch, uint32_t width,
                                       uint32_t has_delta, float swiglu_limit, uint32_t aiv_num);"""),
    # 2. host de-interleave + the two allocations
    ("""    // the kernel takes the two delta halves as separate [T, W] buffers, so
    // de-interleave the reference [T, 2W] host array into two uploads
    std::vector<uint16_t> h_dg(n_out), h_du(n_out);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t j = 0; j < W; j++) {
            h_dg[(size_t)t * W + j] = h_delta[(size_t)t * 2 * W + j];
            h_du[(size_t)t * W + j] = h_delta[(size_t)t * 2 * W + W + j];
        }
    }

    void *d_gu = nullptr, *d_dg = nullptr, *d_du = nullptr, *d_act = nullptr, *d_y = nullptr,
         *d_scale = nullptr;
    ACL_CHECK(aclrtMalloc(&d_gu, n_gu * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_dg, n_out * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_du, n_out * 2, ACL_MEM_MALLOC_HUGE_FIRST));""",
     """    void *d_gu = nullptr, *d_delta = nullptr, *d_act = nullptr, *d_y = nullptr,
         *d_scale = nullptr;
    ACL_CHECK(aclrtMalloc(&d_gu, n_gu * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_delta, n_gu * 2, ACL_MEM_MALLOC_HUGE_FIRST));"""),
    # 3. uploads
    ("""    ACL_CHECK(aclrtMemcpy(d_dg, n_out * 2, h_dg.data(), n_out * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_du, n_out * 2, h_du.data(), n_out * 2, ACL_MEMCPY_HOST_TO_DEVICE));""",
     """    ACL_CHECK(aclrtMemcpy(d_delta, n_gu * 2, h_delta.data(), n_gu * 2, ACL_MEMCPY_HOST_TO_DEVICE));"""),
    # 4. launch
    ("""            vllm_ascend::AscendType::BF16, stream, d_gu, has_delta ? d_dg : nullptr,
            has_delta ? d_du : nullptr, d_act, d_y, d_scale, T, W, has_delta, swiglu_limit,
            aiv_num);""",
     """            vllm_ascend::AscendType::BF16, stream, d_gu, has_delta ? d_delta : nullptr,
            d_act, d_y, d_scale, T, W, has_delta, swiglu_limit,
            aiv_num);"""),
    # 5. frees
    ("""    ACL_CHECK(aclrtFree(d_dg));
    ACL_CHECK(aclrtFree(d_du));""",
     """    ACL_CHECK(aclrtFree(d_delta));"""),
]

for new, old in EDITS:
    if s.count(new) != 1:
        raise SystemExit(f"anchor not found exactly once ({s.count(new)}):\n{new[:120]}")
    s = s.replace(new, old)

open(dst, "w").write(s)
print(f"wrote {dst}")
