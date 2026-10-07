"""Why does origin/test outperform an optimised branch? Check what adding
`hf_to_vllm_mapper` to AscendDeepseekV4ForCausalLM silently switches on.

origin/test = 9fd9bc548 + 2 commits that only refactor weight-name munging into
a WeightsMapper class attribute. That looks inert, but exposing the attribute
flips TWO gates in vLLM that were previously off:

  vllm/model_executor/model_loader/utils.py:296
      if hf_to_vllm_mapper is not None: quant_config.apply_vllm_mapper(...)
          -> rewrites every quant_description KEY (per-layer quant decisions)
  vllm/lora/worker_manager.py:115
      hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
      ... from_local_checkpoint(..., weights_mapper=hf_to_vllm_mapper)
          -> rewrites every LoRA adapter MODULE NAME

Neither fires on 9fd9bc548 or on addlora_path. So a "pure refactor" changes
which weights are quantized and whether adapters attach. This script measures
both, on the real files, with no NPU needed.
"""
import argparse
import glob
import json
import os
import re
import sys

from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.utils import WeightsMapper

# EXACT copy of origin/test vllm_ascend/models/deepseek_v4.py:1223
ORIGIN_TEST_MAPPER = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"rotary_emb\.inv_freq"): None,
        re.compile(r"^(?!model)(?!mtp\.)"): "model.",
    },
    orig_to_new_substr={
        ".w1.": ".gate_proj.",
        ".w2.": ".down_proj.",
        ".w3.": ".up_proj.",
        "model.head.": "lm_head.",
        "model.lm_head.": "lm_head.",
        "embed.": "embed_tokens.",
        ".attn.": ".self_attn.",
        ".ffn.": ".mlp.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".attn_norm.": ".input_layernorm.",
        ".gate.bias": ".gate.e_score_correction_bias",
    },
    orig_to_new_suffix={".scale": ".weight_scale"},
)


def adapter_keys(path):
    from safetensors import safe_open
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="numpy") as fh:
            out.extend(fh.keys())
    return out


def check_lora(adapter):
    print("=" * 100)
    print("LoRA ADAPTER MODULE NAMES:  mapper=None (addlora_path)  vs  mapper (origin/test)")
    print("adapter:", adapter)
    keys = adapter_keys(adapter)
    print(f"tensors: {len(keys)}")
    if keys:
        print("sample raw tensor name:", keys[0])
    changed, same, failed_off, failed_on = 0, 0, [], []
    examples = []
    for k in keys:
        try:
            off, _ = parse_fine_tuned_lora_name(k, None)
        except Exception as e:
            failed_off.append((k, repr(e)))
            off = None
        try:
            on, _ = parse_fine_tuned_lora_name(k, ORIGIN_TEST_MAPPER)
        except Exception as e:
            failed_on.append((k, repr(e)))
            on = None
        if off != on:
            changed += 1
            if len(examples) < 8:
                examples.append((k, off, on))
        else:
            same += 1
    print(f"\nmodule names IDENTICAL : {same}")
    print(f"module names CHANGED   : {changed}   <-- any nonzero means origin/test")
    print(f"                             resolves adapters to DIFFERENT modules")
    print(f"parse errors mapper=None: {len(failed_off)}")
    print(f"parse errors mapper=on  : {len(failed_on)}")
    for k, off, on in examples:
        print(f"\n  raw : {k}")
        print(f"  off : {off}")
        print(f"  on  : {on}")
    for k, e in failed_on[:5]:
        print(f"\n  ERROR with mapper: {k} -> {e}")
    return changed, len(failed_on)


def find_quant_desc(model_dir):
    for pat in ("quant_model_description*.json", "quant_model_description.json",
                "*quant*description*.json"):
        hits = sorted(glob.glob(os.path.join(model_dir, pat)))
        if hits:
            return hits[0]
    return None


def check_quant(model_dir):
    print("=" * 100)
    print("QUANT DESCRIPTION KEYS: unmapped (addlora_path) vs mapped (origin/test)")
    f = find_quant_desc(model_dir)
    if not f:
        print("no quant description json found in", model_dir)
        return None
    print("file:", f)
    qd = json.load(open(f))
    print(f"keys: {len(qd)}")
    mapped = ORIGIN_TEST_MAPPER.apply_dict(dict(qd))
    before, after = set(qd), set(mapped)
    print(f"keys after mapping: {len(mapped)}")
    print(f"  unchanged : {len(before & after)}")
    print(f"  removed   : {len(before - after)}   <-- these keys no longer exist")
    print(f"  added     : {len(after - before)}")
    if len(mapped) != len(qd):
        print(f"  !! KEY COUNT CHANGED by {len(mapped) - len(qd)} -- the mapping COLLIDED,")
        print(f"     so some per-layer quant entries were silently overwritten")
    for k in sorted(before - after)[:6]:
        print(f"   removed: {k}")
    for k in sorted(after - before)[:6]:
        print(f"   added  : {k}")
    return qd, mapped




# ---------------------------------------------------------------------------
# The decisive test, appended: which branch's quant_description actually
# RESOLVES the layer names the vLLM module tree asks for?
#
# get_quant_method() looks up `f"{prefix}.weight" in quant_description`, where
# prefix is the vLLM module prefix (e.g. model.layers.3.self_attn.q_a_proj).
# quant_prefix_mapper() only maps HF->vLLM, so it is a NO-OP on a prefix that is
# already vLLM-form. Therefore the description keys must themselves be vLLM-form
# for the lookup to hit. Run with --hitrate.
# ---------------------------------------------------------------------------

def check_hitrate(model_dir):
    print("=" * 100)
    print("QUANT LOOKUP HIT RATE for the prefixes the vLLM module tree asks for")
    f = find_quant_desc(model_dir)
    qd = json.load(open(f))
    mapped = ORIGIN_TEST_MAPPER.apply_dict(dict(qd))

    # the module prefixes vLLM will query = vLLM-form keys minus the .weight
    prefixes = sorted({k[: -len(".weight")] for k in mapped if k.endswith(".weight")})
    print(f"distinct module prefixes the model will look up: {len(prefixes)}")

    def hits(desc):
        return sum(1 for p in prefixes if f"{p}.weight" in desc)

    h_off, h_on = hits(qd), hits(mapped)
    n = len(prefixes)
    print(f"\n  addlora_path / 9fd9bc548 (quant_description NOT mapped):")
    print(f"      resolved {h_off}/{n} = {100*h_off/n:.2f}%")
    print(f"  origin/test (quant_description mapped):")
    print(f"      resolved {h_on}/{n} = {100*h_on/n:.2f}%")
    missed = [p for p in prefixes if f"{p}.weight" not in qd][:10]
    print("\n  examples the UNMAPPED description cannot resolve:")
    for p in missed:
        print("     ", p)
    print("\n  INTERPRETATION: a prefix that does not resolve gets no quant scheme,")
    print("  so that layer is built UNQUANTIZED (bf16) instead of W8A8. Fewer")
    print("  resolved prefixes => more bf16 layers => slower and more memory.")
    return h_off, h_on, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16")
    ap.add_argument("--model", default="/data/models/DeepSeek-V4-Flash-0731-w8a8")
    a = ap.parse_args()
    check_lora(a.adapter)
    print()
    check_quant(a.model)
    print()
    check_hitrate(a.model)
    return 0



if __name__ == "__main__":
    sys.exit(main())
