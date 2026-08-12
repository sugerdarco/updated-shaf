# Byte Prefix Tree Construction & Execution Guide

This folder contains the complete, self-contained implementation of the **Byte Prefix Tree (Shared Base Space)** used in the Stage 8 cross-tokenizer (SHEAF) multi-agent ensemble architecture.

---

## 1. Execution Timing: Before vs. After Pipeline

The Byte Prefix Tree is involved at **two distinct times**:

```
                       [ BEFORE INFERENCE RUNS ]
                                   │
             ┌─────────────────────┴─────────────────────┐
             │ Step 0: Offline Tree Construction         │
             │ Built ONCE offline via build_prefix_tree  │
             │ or when the pipeline initializes.         │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
                       [ AT EVERY DECODING STEP ]
                                   │
┌──────────────────────────────────┴──────────────────────────────────┐
│ Step A: Probability Mass Propagation                                │
│ Agent logits pushed into Prefix Tree BEFORE Stage 5/6/7 Fusion.     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────┴──────────────────────────────────┐
│ Step B: Stage 5/6/7 Fusion & Escalation                             │
│ Reconciler runs per tree node (Mean Fusion -> Outliers -> GM).      │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Tree Construction (Runs BEFORE Inference)**: 
   * The tree geometry depends *only* on the models' token vocabularies, **not** on any prompt or input text. 
   * It is built **once before running generation** via `python build_prefix_tree.py` and saved as an artifact (`prefix_tree.npz`).
2. **Mass Propagation (Runs at the START of every decoding step)**:
   * When an agent produces its next-token probability distribution $P_i(v)$, those probabilities are pushed **into** the pre-built prefix tree **before** Stage 5 (Fusion), Stage 6 (Outliers), and Stage 7 (Geometric Median Escalation) can run.

---

## 2. Steps to Run

To build the Prefix Tree artifact once before running inference:

```bash
# Build the prefix tree artifact once (CPU-only, no GPU or model weights required)
python build_prefix_tree.py --config config_sheaf.yaml
```

This generates `artifacts/prefix_tree.npz`, which is loaded during decoding.

---

## 3. Five Internal Steps Inside the Byte Prefix Tree

Inside `prefix_tree.py` and `vocab.py`, 5 main steps take place:

### Step 1: Raw Byte Extraction (`vocab.py`)
Each token in each agent's vocabulary is converted to its raw byte representation (e.g., Llama token `29892` $\rightarrow$ `b","`, Qwen token `151643` $\rightarrow$ `b"<|endoftext|>"`). Supported schemes:
* **Byte-level BPE** (GPT-2, Llama-3, Qwen, Mistral-v3)
* **SentencePiece** (Llama-2, Gemma, T5)
* **WordPiece** (BERT)

### Step 2: Union Trie Construction (`prefix_tree.py`)
Creates a unified byte tree (Trie) over all agent vocabularies:
* Root node ($s = \text{empty string } b""$).
* Each edge represents one byte value ($0 \dots 255$).
* A node at depth $d$ represents a byte prefix of length $d$ (e.g. $b\text{"P"} \rightarrow b\text{"Pa"} \rightarrow b\text{"Par"} \rightarrow b\text{"Paris"}$).

### Step 3: Pre-computed Path Indexing (`prefix_tree.py`)
For every token in every model, its path from root to leaf node is pre-computed into flat `numpy` arrays. This allows mass propagation to run at matrix speed without Python loops.

### Step 4: Dual Mass Calculation (`prefix_tree.py`)
For every node $s$ in the tree and each agent $a$:
* **`cover(s)`**: Total probability mass of all tokens whose byte representation **starts with** $s$.
* **`term(s)`**: Total probability mass of tokens whose byte representation **equals exactly** $s$.

### Step 5: Simplex Standardization for Fusion (`prefix_tree.py`)
The continuing mass at node $s$ is calculated as:
$$\text{continuation}(s) = \text{cover}(s) - \text{term}(s)$$
This maps all agents' distributions onto a shared 256-dimensional next-byte probability simplex, allowing Stages 5, 6, and 7 to perform amplitude fusion and outlier rejection seamlessly across mismatched tokenizers.

---

## 4. File Structure

```
prefix_tree_build/
├── README.md               # Detailed documentation & extracted 5-step breakdown
├── __init__.py             # Module exports
├── vocab.py                # Token-to-byte extraction & scheme detection
├── prefix_tree.py          # Union BytePrefixTree data structure & propagation
├── build_prefix_tree.py    # Standalone CLI builder script
└── config_sheaf.yaml       # Configuration file for model tokenizers
```

* [`README.md`](README.md): This documentation file.
* [`vocab.py`](vocab.py): Vocabulary extraction, scheme detection, and byte decoding.
* [`prefix_tree.py`](prefix_tree.py): Union byte-prefix tree data structure & fast mass propagation.
* [`build_prefix_tree.py`](build_prefix_tree.py): CLI tool to precompute and save prefix tree artifacts.
* [`config_sheaf.yaml`](config_sheaf.yaml): YAML configuration for models and tree paths.
