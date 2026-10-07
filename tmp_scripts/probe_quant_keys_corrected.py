"""Corrected comparison of the EFFECTIVE quant_description on each branch.

An earlier version of this probe compared the mapper against the RAW
quant_model_description.json and reported 0.02% vs 100% resolution. That was
WRONG as a branch comparison: AscendModelSlimConfig.__init__ calls
_apply_extra_quant_adaptations(), which (when "hc_head_fn" is present, and it is
for this checkpoint) performs the same HF->vLLM renaming and merges it in with
dict.update() -- building a UNION that keeps BOTH key forms. That is why
quantization works on addlora_path.

The real structural difference:

  addlora_path / 9fd9bc548 : quant_description = raw UNION vllm-form   (both)
  origin/test              : apply_dict(raw UNION vllm-form)           (vllm-form
                             only -- apply_dict REPLACES keys, so the HF-form
                             keys are GONE)

So any lookup that relies on an HF-form key succeeds on addlora_path and fails
on origin/test. This script measures how many such keys there are.
"""
import json
import re
import sys

from vllm.model_executor.models.utils import WeightsMapper

ORIGIN_TEST_MAPPER = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"rotary_emb\.inv_freq"): None,
        re.compile(r"^(?!model)(?!mtp\.)"): "model.",
    },
    orig_to_new_substr={
        ".w1.": ".gate_proj.", ".w2.": ".down_proj.", ".w3.": ".up_proj.",
        "model.head.": "lm_head.", "model.lm_head.": "lm_head.",
        "embed.": "embed_tokens.", ".attn.": ".self_attn.", ".ffn.": ".mlp.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".attn_norm.": ".input_layernorm.",
        ".gate.bias": ".gate.e_score_correction_bias",
    },
    orig_to_new_suffix={".scale": ".weight_scale"},
)


def adaptations(qd):
    """Faithful replay of _apply_extra_quant_adaptations (modelslim_config.py:920)."""
    qd = dict(qd)
    if "hc_head_fn" in qd:
        for rename in (
            lambda n: f"model.{n}" if not n.startswith("model") else n,
            lambda n: n.replace(".attn.", ".self_attn.") if ("attn" in n and "self_attn" not in n) else n,
            lambda n: n.replace("ffn", "mlp") if "ffn" in n else n,
        ):
            extra = {rename(n): v for n, v in qd.items()}
            qd.update(extra)
        extra = {}
        for n, v in qd.items():
            nn = n
            if "w1" in n: nn = n.replace(".w1.", ".gate_proj.")
            if "w2" in n: nn = n.replace(".w2.", ".down_proj.")
            if "w3" in n: nn = n.replace(".w3.", ".up_proj.")
            if "head" in n and "lm_head" not in n: nn = n.replace("head", "lm_head")
            if "embed" in n and "embed_tokens" not in n: nn = n.replace("embed", "embed_tokens")
            extra[nn] = v
        qd.update(extra)
    return qd


def main(path):
    raw = json.load(open(path))
    eff_addlora = adaptations(raw)                              # the union
    eff_test = ORIGIN_TEST_MAPPER.apply_dict(dict(eff_addlora))  # then mapped

    print(f"raw json keys                      : {len(raw)}")
    print(f"addlora_path effective (union)     : {len(eff_addlora)}")
    print(f"origin/test effective (union+map)  : {len(eff_test)}")

    a, b = set(eff_addlora), set(eff_test)
    lost = a - b
    gained = b - a
    print(f"\nkeys addlora_path HAS and origin/test LOST : {len(lost)}")
    print(f"keys only origin/test has                  : {len(gained)}")
    if len(eff_test) < len(eff_addlora):
        print(f"\n!! origin/test's description is SMALLER by {len(eff_addlora)-len(eff_test)} keys.")
        print("   apply_dict REPLACES keys, so HF-form entries collapsed onto their")
        print("   vLLM-form twins. Harmless only if the collapsed pairs held equal")
        print("   values -- checked below.")

    # did any collapse change a value?
    conflicts = 0
    for k in lost:
        mk = ORIGIN_TEST_MAPPER._map_name(k)
        if mk is None:
            continue
        if mk in eff_test and eff_test[mk] != eff_addlora[k]:
            conflicts += 1
            if conflicts <= 5:
                print(f"   VALUE CONFLICT {k} ({eff_addlora[k]}) -> {mk} ({eff_test[mk]})")
    print(f"\ncollapses that CHANGED a quant value: {conflicts}")

    # which forms can each resolve?
    prefixes_vllm = sorted({k[:-len('.weight')] for k in eff_test if k.endswith('.weight')})
    def hit(d, ps): return sum(1 for p in ps if f"{p}.weight" in d)
    n = len(prefixes_vllm)
    print(f"\nvLLM-form prefixes ({n}):")
    print(f"   addlora_path resolves {hit(eff_addlora,prefixes_vllm)}/{n}")
    print(f"   origin/test  resolves {hit(eff_test,prefixes_vllm)}/{n}")
    hf_only = sorted({k[:-len('.weight')] for k in lost if k.endswith('.weight')})
    print(f"\nHF-form prefixes that ONLY addlora_path can resolve: {len(hf_only)}")
    for p in hf_only[:8]:
        print("   ", p)
    mtp = [k for k in lost if k.startswith("mtp.")]
    print(f"\nof those, MTP-related keys lost by origin/test: {len(mtp)}")
    for k in mtp[:8]:
        print("   ", k)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/data/models/DeepSeek-V4-Flash-0731-w8a8/quant_model_description.json")
