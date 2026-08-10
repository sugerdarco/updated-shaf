"""Audit script for the standalone Prefix Tree module.

Adversarial probes checking mathematical invariants and tokenization edge cases
specifically for Vocabulary Extraction and Prefix Tree construction.
"""

import sys
from pathlib import Path

import numpy as np

# Point to our standalone prefix_tree_build folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prefix_tree_build"))

from prefix_tree import BytePrefixTree
from vocab import VocabSpec, bytes_to_unicode, detect_scheme

FAILS = []


def check(stage, name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(f"[{stage}] {name}: {detail}")
    print(f"  {tag}  {name}" + (f"   -- {detail}" if detail and not ok else ""))


def vocab(name, toks):
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(n, w):
    p = np.zeros(n)
    for i, v in w.items():
        p[i] = v
    return p / p.sum()


# =====================================================================
print("\nSTAGE A -- vocabulary extraction (vocab.py)")
# =====================================================================

enc = bytes_to_unicode()
ok = all("".join(enc[b] for b in bytes([i])) and True for i in range(256))
from vocab import decode_bytelevel

roundtrip = all(decode_bytelevel(enc[i]) == bytes([i]) for i in range(256))
check("A", "byte-level covers all 256 byte values", roundtrip)

# A1: byte-level vocab that legitimately contains "##" (markdown/code corpora)
bl_vocab = ["".join(enc[b] for b in t) for t in [b"##", b" the", b"\n", b"def", b" x"]]
scheme = detect_scheme(bl_vocab)
check(
    "A",
    "byte-level vocab containing '##' is not misread as wordpiece",
    scheme == "byte_level",
    f"detected {scheme!r}, expected 'byte_level'",
)

# A2: genuine wordpiece must still be detected
wp_vocab = ["play", "##ing", "##ed", "the", "cat", "[UNK]"]
scheme = detect_scheme(wp_vocab)
check("A", "genuine wordpiece still detected", scheme == "wordpiece", f"detected {scheme!r}")

# A3: sentencepiece
sp_vocab = ["\u2581the", "\u2581cat", "<0x0A>", "s"]
check("A", "sentencepiece detected", detect_scheme(sp_vocab) == "sentencepiece")

# A4: control / added tokens leaking into the byte space
chat_vocab = ["".join(enc[b] for b in t) for t in [b"<|im_start|>", b" hi"]]


class FakeTok:
    def __init__(self, toks):
        self._v = {t: i for i, t in enumerate(toks)}
        self.all_special_ids = []
        self.vocab_size = len(toks)
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._v

    def encode(self, text, add_special_tokens=False):
        return []


vs = VocabSpec.from_hf(FakeTok(chat_vocab), scheme="byte_level")
leaks = vs.token_bytes[0] == b"<|im_start|>"
check(
    "A",
    "control tokens do not enter the byte tree as literal text",
    not leaks,
    "'<|im_start|>' decoded to its literal bytes and will be fused as real text",
)

# =====================================================================
print("\nSTAGE B -- byte-prefix tree (prefix_tree.py)")
# =====================================================================

v = vocab("a", [b" the", b" th", b" cat", b"x"])
tree = BytePrefixTree.from_vocabs([v])
p = dist(4, {0: 0.4, 1: 0.2, 2: 0.3, 3: 0.1})
cover, term = tree.cover_mass(0, p)
ident = all(
    abs(cover[n] - (term[n] + sum(cover[c] for c in tree.children[n].values()))) < 1e-12
    for n in range(tree.n_nodes)
)
check("B", "cover(s) = term(s) + sum children", ident)
check("B", "root cover == 1", abs(cover[0] - 1.0) < 1e-12)

# B1: probability vector shorter than the vocab table
short = np.array([0.5, 0.5])
try:
    tree.cover_mass(0, short)
    ok, detail = False, "silently accepted a mis-sized probability vector"
except IndexError:
    ok, detail = False, "raised a bare IndexError instead of a clear error"
except ValueError as e:
    ok, detail = True, str(e)
check("B", "mis-sized probability vector rejected clearly", ok, detail)

# B2: duplicate ids in restrict
tree_dup = BytePrefixTree.from_vocabs([v], restrict=[[0, 0, 1]])
cov_d, _ = tree_dup.cover_mass(0, p)
check(
    "B",
    "duplicate ids in restrict do not double-count mass",
    abs(cov_d[0] - 0.6) < 1e-12,
    f"root cover = {cov_d[0]:.4f}, expected 0.6 (0.4+0.2)",
)

# B3: truncation preserves the identity
v_long = vocab("a", [b"abcdef", b"abcxyz", b"ab"])
t_trunc = BytePrefixTree.from_vocabs([v_long], max_depth=3)
c2, t2 = t_trunc.cover_mass(0, dist(3, {0: 0.5, 1: 0.3, 2: 0.2}))
ident2 = all(
    abs(c2[n] - (t2[n] + sum(c2[c] for c in t_trunc.children[n].values()))) < 1e-12
    for n in range(t_trunc.n_nodes)
)
check("B", "identity holds under max_depth truncation", ident2)

# B4: monotonicity (parent >= child), required for the depth-sorted glue walk
mono = all(
    cover[n] >= cover[c] - 1e-15 for n in range(tree.n_nodes) for c in tree.children[n].values()
)
check("B", "cover mass is monotone down the tree", mono)

if FAILS:
    print(f"\n{len(FAILS)} checks failed.")
    sys.exit(1)
else:
    print("\nAll checks passed.")
