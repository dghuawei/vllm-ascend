import json, sys

# usage: compare_outputs.py outsA.json outsB.json
# Prints: lora-vs-base within each state (applied? applied-different?)
# and A_lora vs B_lora (branch-to-branch identity).
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

def get(d, m, i):
    return d.get(f"{m}|{i}")

idx = sorted({int(k.split("|")[1]) for k in a})
same_al, diff_al = [], []
same_ab, diff_ab = [], []
for i in idx:
    a_l = get(a, "lora-adapter1", i)
    a_b = get(a, "ds", i)
    b_l = get(b, "lora-adapter1", i)
    if a_l == a_b:
        diff_al.append(i)  # addlora lora == base -> not applied (state A)
    else:
        same_al.append(i)
    if a_l == b_l:
        same_ab.append(i)
    else:
        diff_ab.append(i)

print(f"stateA(addlora): lora differs from base on {len(same_al)}/{len(idx)} prompts "
      f"(equal-to-base: {diff_al})")
print(f"A_lora == B_lora on {len(same_ab)}/{len(idx)} prompts (diverged: {diff_ab})")
for i in (diff_ab[:2] + diff_al[:2]):
    print(f"--- prompt {i} ---")
    print("A_lora:", repr(get(a, 'lora-adapter1', i))[:180])
    print("A_base:", repr(get(a, 'ds', i))[:180])
    print("B_lora:", repr(get(b, 'lora-adapter1', i))[:180])
