"""
Build the Stage 8 byte-prefix tree ONCE and save it as a reusable artifact.

Usage:
    python build_prefix_tree.py                              # uses config_sheaf.yaml
    python build_prefix_tree.py --config config_sheaf.yaml --out artifacts/prefix_tree.npz

Only TOKENIZERS are loaded here — no model weights, no GPU needed.
"""

import argparse
import time
from pathlib import Path

import yaml

try:
    from .vocab import VocabSpec
    from .prefix_tree import BytePrefixTree
except ImportError:
    from vocab import VocabSpec
    from prefix_tree import BytePrefixTree

VERIFY_SAMPLES = [
    "The capital of France is Paris",
    "the cat sat on the mat",
    "def main():\n    return 1",
]


def main():
    parser = argparse.ArgumentParser(description="Build Union Byte-Prefix Tree")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_sheaf.yaml"))
    parser.add_argument("--out", default=None, help="Artifact path (default: from config).")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Truncate tokens longer than this many bytes. Omit for exact.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the byte round-trip check. Not recommended.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path(args.out or cfg.get("sheaf", {}).get("tree_path", "artifacts/prefix_tree.npz"))

    from transformers import AutoTokenizer

    specs = []
    for name in cfg["models"]:
        print(f"loading tokenizer: {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        vs = VocabSpec.from_hf(tok, name=name)
        print(f"  scheme={vs.scheme}  vocab={vs.size}")

        if not args.skip_verify:
            if not vs.verify(tok, VERIFY_SAMPLES):
                raise SystemExit(
                    f"Byte round-trip FAILED for {name} (scheme={vs.scheme}). "
                    "The tree would be silently misaligned; fix extraction first."
                )
        specs.append(vs)

    print("\nbuilding union byte-prefix tree ...", flush=True)
    t0 = time.perf_counter()
    tree = BytePrefixTree.from_vocabs(specs, max_depth=args.max_depth)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree.save(out)
    save_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    reloaded = BytePrefixTree.load(out)
    load_s = time.perf_counter() - t0
    assert reloaded.n_nodes == tree.n_nodes, "artifact did not round-trip"

    size_mb = out.stat().st_size / 1e6
    print(
        f"\n  nodes      : {tree.n_nodes:,}\n"
        f"  max depth  : {tree.max_depth_seen}\n"
        f"  agents     : {tree.n_agents}\n"
        f"  build      : {build_s:.1f} s\n"
        f"  save       : {save_s:.1f} s\n"
        f"  reload     : {load_s:.1f} s\n"
        f"  artifact   : {out} ({size_mb:.1f} MB)\n"
    )


if __name__ == "__main__":
    main()
