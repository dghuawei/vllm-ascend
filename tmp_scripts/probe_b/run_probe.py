"""Build and run the strided-output probe for aclnnGroupedMatmulV4."""
import os
import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load

TN = os.path.dirname(torch_npu.__file__)
A = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
SRC = os.path.dirname(os.path.abspath(__file__))

mod = load(
    name="probe_b",
    sources=[
        os.path.join(SRC, "probe.cpp"),
        os.path.join(SRC, "adapter", "NPUBridge.cpp"),
        os.path.join(SRC, "adapter", "NPUStorageImpl.cpp"),
    ],
    extra_include_paths=[
        os.path.join(SRC, "adapter"),
        os.path.join(TN, "include"),
        os.path.join(TN, "include/third_party/op-plugin/op_plugin/utils"),
        os.path.join(A, "include"),
    ],
    extra_cflags=["-std=c++17"],
    extra_ldflags=[f"-L{TN}/lib", "-ltorch_npu", f"-L{A}/lib64", "-lascendcl", "-lnnopbase", "-lopapi"],
    verbose=True,
)

torch.npu.set_device(0)
T, R, N, E = 64, 16, 128, 4
dt = torch.bfloat16
x = torch.randn(T, R, device="npu", dtype=dt)
w = torch.randn(E, R, N, device="npu", dtype=dt)
gl = torch.tensor([16, 16, 16, 16], device="npu", dtype=torch.int64)

ref = mod.gmm_contig(x, w, gl)
torch.npu.synchronize()

x2 = torch.randn(T, R, device="npu", dtype=dt)
w2 = torch.randn(E, R, N, device="npu", dtype=dt)
ref2 = mod.gmm_contig(x2, w2, gl)
torch.npu.synchronize()

dst = torch.zeros(T, 2 * N, device="npu", dtype=dt)
mod.gmm_strided(x, w, gl, dst, 0)      # slice 0 -> columns [0, N)
mod.gmm_strided(x2, w2, gl, dst, N)    # slice 1 -> columns [N, 2N), storage_offset != 0
torch.npu.synchronize()

left, right = dst[:, :N], dst[:, N:]
ok_l = torch.equal(left.float(), ref.float())
ok_r = torch.equal(right.float(), ref2.float())
print(f"slice 0 (offset 0)  : {ok_l}  max|diff|={(left.float()-ref.float()).abs().max().item():.6g}")
print(f"slice 1 (offset {N}) : {ok_r}  max|diff|={(right.float()-ref2.float()).abs().max().item():.6g}")
print("VERDICT:", "B WORKS" if ok_l and ok_r else "B FAILS")
