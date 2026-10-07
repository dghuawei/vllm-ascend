#!/usr/bin/env python3
"""Truncate a DeepSeek-V4-Flash w8a8 checkpoint to its first N transformer layers.

Produced for the b3-vs-b4 apples-to-apples profiling: the bz-ascend node only
has a 6-layer truncated export; this builds the same shape of checkpoint from
the full one on researchagent-node1.

What is kept:
  * layers.0..N-1.*                    (transformer layers incl. experts/scales)
  * embed.weight, head.weight, norm.weight, hc_head_{base,fn,scale}
  * everything in tokenizer/chat/generation config files
What is dropped:
  * layers.N.. and the whole mtp.* block (profiling runs with MTP disabled)
Rewritten:
  * quant_model_weights.safetensors.index.json  (new shards + index)
  * quant_model_description.json  (filtered to kept tensor names)
  * config.json  (num_hidden_layers, compress_ratios prefix, MTP off)
"""
import argparse
import json
import os
import re
import shutil
import time

from safetensors import safe_open
from safetensors.torch import save_file

SHARD_NAME = "quant_model_weights-{:05d}-of-{:05d}.safetensors"
INDEX_NAME = "quant_model_weights.safetensors.index.json"
NON_LAYER_KEEP = {
    "embed.weight",
    "head.weight",
    "norm.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
}
COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
]
LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument(
        "--compress-ratios",
        default=None,
        help="comma list overriding config.json compress_ratios "
        "(must match the reference truncated export; default = first N of the source)",
    )
    ap.add_argument(
        "--max-shard-gb",
        type=float,
        default=4.0,
        help="target output shard size; the 43L source uses ~4.2 GB/shard",
    )
    args = ap.parse_args()

    src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
    os.makedirs(dst, exist_ok=True)

    index = json.load(open(os.path.join(src, INDEX_NAME)))
    wmap = index["weight_map"]

    keep, dropped = [], []
    for k in wmap:
        m = LAYER_RE.match(k)
        if m:
            (keep if int(m.group(1)) < args.layers else dropped).append(k)
        elif k.startswith("mtp."):
            dropped.append(k)
        elif k in NON_LAYER_KEEP:
            keep.append(k)
        else:
            dropped.append(k)
    keep = sorted(keep)
    print(f"keys: keep={len(keep)} drop={len(dropped)} (source total {len(wmap)})")

    # read tensors shard-grouped so each source file is opened once
    by_shard = {}
    for k in keep:
        by_shard.setdefault(wmap[k], []).append(k)

    max_bytes = int(args.max_shard_gb * (1 << 30))
    tensors_out, new_wmap, total_size, cur_bytes = {}, {}, 0, 0
    t0 = time.time()

    def flush():
        nonlocal tensors_out, cur_bytes
        if not tensors_out:
            return
        n = len(out_shards) + 1
        out_shards.append(n)
        # shard count is unknown until the end; rename in a final pass
        fname = f"pending_{n:05d}.safetensors"
        save_file(tensors_out, os.path.join(dst, fname), metadata={"format": "pt"})
        for k in tensors_out:
            new_wmap[k] = fname
        print(
            f"  wrote {fname}: {len(tensors_out)} keys, "
            f"{cur_bytes/2**30:.2f} GiB ({time.time()-t0:.0f}s elapsed)"
        )
        tensors_out, cur_bytes = {}, 0

    out_shards = []
    for shard in sorted(by_shard):
        keys = sorted(by_shard[shard])
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                nbytes = t.numel() * t.element_size()
                if cur_bytes + nbytes > max_bytes and tensors_out:
                    flush()
                tensors_out[k] = t
                cur_bytes += nbytes
                total_size += nbytes
        print(f"  read {shard} ({time.time()-t0:.0f}s)")
    flush()

    # final rename to -of-TOTAL + real index
    n_total = len(out_shards)
    rename = {}
    for n in out_shards:
        old = f"pending_{n:05d}.safetensors"
        new = SHARD_NAME.format(n, n_total)
        os.rename(os.path.join(dst, old), os.path.join(dst, new))
        rename[old] = new
    for k in new_wmap:
        new_wmap[k] = rename[new_wmap[k]]
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": new_wmap},
        open(os.path.join(dst, INDEX_NAME), "w"),
        indent=2,
    )
    print(f"index written: {n_total} shards, total_size={total_size/1e9:.2f} GB")

    # quant_model_description.json -> filter to kept names (tolerate absence)
    qd_path = os.path.join(src, "quant_model_description.json")
    if os.path.exists(qd_path):
        qd = json.load(open(qd_path))
        qd_new = {k: v for k, v in qd.items() if k in set(keep)}
        json.dump(qd_new, open(os.path.join(dst, "quant_model_description.json"), "w"))
        print(f"quant_model_description: {len(qd)} -> {len(qd_new)} entries")

    # config.json
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["num_hidden_layers"] = args.layers
    if args.compress_ratios:
        cfg["compress_ratios"] = [int(x) for x in args.compress_ratios.split(",")]
    else:
        cfg["compress_ratios"] = cfg["compress_ratios"][: args.layers]
    cfg["num_nextn_predict_layers"] = 0
    cfg["dspark_target_layer_ids"] = []
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"config.json: layers={args.layers} compress_ratios={cfg['compress_ratios']}")

    for f in COPY_FILES:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, f))
            print(f"copied {f}")

    # consistency: every kept key present exactly once in the new index
    assert set(new_wmap) == set(keep), "index/key mismatch"
    print("DONE ->", dst)


if __name__ == "__main__":
    main()
