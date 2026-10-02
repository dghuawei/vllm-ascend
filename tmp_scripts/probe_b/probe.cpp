// Does aclnnGroupedMatmulV4 honour a NON-CONTIGUOUS (strided) output tensor?
// Runs the same grouped matmul twice: once into a fresh contiguous buffer,
// once into a [T, 2N] buffer's left half (row stride 2N). If the strided run
// matches the contiguous one, option B is viable.
#include <torch/extension.h>
#include "op_api_common.h"

namespace {
at::Tensor gmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &group_list,
               const c10::optional<at::Tensor> &out_opt)
{
    int64_t T = x.size(0);
    int64_t N = w.size(w.dim() - 1);
    at::Tensor out = out_opt.has_value() ? out_opt.value() : at::empty({T, N}, x.options());
    std::vector<at::Tensor> xv{x}, wv{w}, ov{out};
    at::TensorList xs(xv), ws(wv), result(ov);
    at::TensorList none, act_out, dyn;
    int64_t split_item = 2, group_type = 0, group_list_type = 1, act_type = 0;
    EXEC_NPU_CMD(aclnnGroupedMatmulV4, xs, ws, none, none, none, none,
                 none, none, group_list, none,
                 none, none, split_item, group_type,
                 group_list_type, act_type, result, act_out, dyn);
    return out;
}
}  // namespace

at::Tensor gmm_contig(at::Tensor x, at::Tensor w, at::Tensor gl)
{
    return gmm(x, w, gl, c10::nullopt);
}

at::Tensor gmm_strided(at::Tensor x, at::Tensor w, at::Tensor gl, at::Tensor dst, int64_t off)
{
    int64_t N = w.size(w.dim() - 1);
    at::Tensor view = dst.slice(1, off, off + N);
    TORCH_CHECK(!view.is_contiguous(), "probe: view unexpectedly contiguous");
    gmm(x, w, gl, view);
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gmm_contig", &gmm_contig);
    m.def("gmm_strided", &gmm_strided);
}
