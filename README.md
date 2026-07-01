# 🔁 Transformer Encoder–Decoder: Step-by-Step Numeric Walkthrough

> A fully worked, numeric trace of the original Transformer (Vaswani et al., 2017) architecture — from raw text to predicted token — using a translation example:
> **English → Tamil**: `"I am fine"` → `"நான் நலம்"`

<p align="center">
  <img src="https://img.shields.io/badge/architecture-Transformer-blueviolet" />
  <img src="https://img.shields.io/badge/task-Machine%20Translation-orange" />
  <img src="https://img.shields.io/badge/type-Encoder--Decoder-brightgreen" />
</p>

---

## ⚙️ Toy Dimensions Used

| Symbol | Value | Meaning |
|---|---|---|
| `d_model` | 4 | embedding dimension |
| `num_heads` | 2 | number of attention heads |
| `d_k` | 2 | dimension per head (`d_model / num_heads`) |
| `vocab_size` | 6 | vocabulary size |

> Real models use `d_model = 512`, `num_heads = 8`, `d_k = 64`, `vocab_size = 30k–50k` — same formulas, bigger numbers.

---

## 📑 Table of Contents

- [Part 1 — Encoder Pipeline](#part-1--encoder-pipeline)
  - [Step 1 — Input Sentence](#step-1--input-sentence)
  - [Step 2 — Tokenization (BPE)](#step-2--tokenization-bpe)
  - [Step 3 — Input Embedding](#step-3--input-embedding)
  - [Step 4 — Positional Encoding](#step-4--positional-encoding)
  - [Step 5 — Linear Projection into Q, K, V](#step-5--linear-projection-into-q-k-v)
  - [Step 6 — Split into Heads](#step-6--split-into-heads)
  - [Step 7 — MatMul: Q × Kᵀ](#step-7--matmul-q--kᵀ)
  - [Step 8 — Scale](#step-8--scale)
  - [Step 9 — Softmax (row-wise)](#step-9--softmax-row-wise)
  - [Step 10 — MatMul: weights × V](#step-10--matmul-weights--v)
  - [Step 11 — Concatenate Heads](#step-11--concatenate-heads)
  - [Step 12 — Final Linear Projection (W_O)](#step-12--final-linear-projection-w_o)
  - [Step 13 — Add & Norm](#step-13--add--norm-residual--layernorm)
  - [Step 14 — Feed Forward NN](#step-14--feed-forward-nn)
  - [Step 15 — Add & Norm → Encoder Output](#step-15--add--norm--encoder-output)
- [Part 2 — Decoder Pipeline](#part-2--decoder-pipeline)
  - [Step 16 — Target Input (shifted right)](#step-16--target-input-shifted-right)
  - [Step 17 — Output Embedding](#step-17--output-embedding)
  - [Step 18 — + Positional Encoding](#step-18---positional-encoding)
  - [Step 19 — Q, K, V Projection (decoder's own)](#step-19--q-k-v-projection-decoders-own)
  - [Step 20 — MatMul Q×Kᵀ, then Scale](#step-20--matmul-qkᵀ-then-scale)
  - [Step 21 — Masking](#step-21--masking-the-extra-step-vs-encoder)
  - [Step 22 — Softmax](#step-22--softmax)
  - [Step 23 — MatMul weights × V, Concat, Final Linear](#step-23--matmul-weights--v-concat-final-linear)
  - [Step 24 — Add & Norm](#step-24--add--norm)
  - [Step 25 — Cross-Attention](#step-25--cross-attention)
  - [Step 26 — Add & Norm](#step-26--add--norm-1)
  - [Step 27 — Feed Forward NN](#step-27--feed-forward-nn)
  - [Step 28 — Add & Norm → Final Decoder Output](#step-28--add--norm--final-decoder-output)
  - [Step 29 — Linear Layer](#step-29--linear-layer)
  - [Step 30 — Softmax](#step-30--softmax-1)
  - [Step 31 — Predicted Word (Final Output)](#step-31--predicted-word-final-output)
- [📋 Quick Formula Reference Sheet](#-quick-formula-reference-sheet)
- [🗺️ Full Pipeline at a Glance](#️-full-pipeline-at-a-glance)

---

## Part 1 — Encoder Pipeline

### Step 1 — Input Sentence

| | |
|---|---|
| **Input** | `"I am fine"` |
| **Output** | Raw text string, unchanged, sent to tokenizer |

---

### Step 2 — Tokenization (BPE)

| | |
|---|---|
| **Input** | `"I am fine"` |
| **Process** | Byte-Pair merge of frequent sub-word pairs |
| **Output** | `[I]=12, [am]=45, [fine]=78` → `[12, 45, 78]` |
| **Shape** | `(seq_len=3,)` |

---

### Step 3 — Input Embedding

**Input:** `[12, 45, 78]`

**Formula** — each token ID is looked up in embedding table `E ∈ ℝ^(vocab_size × d_model)`:

```
x_i = E[token_id_i]
```

**Output (toy example, d_model=4):**

```
x1 (I)    = [0.10, 0.20, 0.05, 0.30]
x2 (am)   = [0.15, 0.10, 0.25, 0.05]
x3 (fine) = [0.30, 0.05, 0.10, 0.20]
```

**Shape:** `(seq_len=3, d_model=4)`

---

### Step 4 — Positional Encoding

**Input:** Embedding matrix `(3, 4)`

**Formula:**

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

where `pos` = position of word in sentence (0,1,2,...), `i` = dimension index pair.

<details>
<summary><b>🔢 Worked calculation for pos=1 ("am"), d_model=4</b></summary>

```
dim 0 (2i=0,   i=0): sin(1 / 10000^(0/4)) = sin(1/1)     = sin(1)    = 0.8415
dim 1 (2i+1=1, i=0): cos(1 / 10000^(0/4)) = cos(1/1)     = cos(1)    = 0.5403
dim 2 (2i=2,   i=1): sin(1 / 10000^(2/4)) = sin(1/100)   = sin(0.01) = 0.0100
dim 3 (2i+1=3, i=1): cos(1 / 10000^(2/4)) = cos(1/100)   = cos(0.01) = 0.9999
```

`PE(pos=1) = [0.8415, 0.5403, 0.0100, 0.9999]`

</details>

**Add to embedding:**

```
final_input = x_i + PE(pos_i)
```

**Output:** `(seq_len=3, d_model=4)` — position-aware embeddings

---

### Step 5 — Linear Projection into Q, K, V

**Input:** `X (3, 4)` (from Step 4)

**Formula:**

```
Q = X · W_Q      W_Q ∈ ℝ^(d_model × d_model)
K = X · W_K      W_K ∈ ℝ^(d_model × d_model)
V = X · W_V      W_V ∈ ℝ^(d_model × d_model)
```

**Output:** Q, K, V each `(3, 4)`

---

### Step 6 — Split into Heads

| | |
|---|---|
| **Input** | Q, K, V each `(3, 4)` |
| **Process** | reshape `d_model=4` into `num_heads=2 × d_k=2` |
| **Output** | Q, K, V each `(heads=2, seq_len=3, d_k=2)` |

---

### Step 7 — MatMul: Q × Kᵀ

| | |
|---|---|
| **Input** | Q `(3, 2)`, K `(3, 2)` per head |
| **Formula** | `scores = Q · Kᵀ` |
| **Output** | raw score matrix `(3, 3)` per head — every word's raw score vs every other word |

---

### Step 8 — Scale

**Input:** `scores (3, 3)`

**Formula:**

```
scaled_scores = scores / √d_k     ( √2 ≈ 1.414 in toy example, √64 in real model )
```

**Output:** `(3, 3)`, same shape, smaller magnitude (keeps gradients stable)

---

### Step 9 — Softmax (row-wise)

**Input:** `scaled_scores (3, 3)`

**Formula:**

```
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**Example output for row "am" attending to `[I, am, fine]`:**

```
attention_weights("am") = [0.15, 0.70, 0.15]   (sums to 1.0)
```

→ "am" pays 70% attention to itself, 15% each to "I" and "fine".

**Output:** `(3, 3)` attention weight matrix per head

---

### Step 10 — MatMul: weights × V

| | |
|---|---|
| **Input** | attention weights `(3, 3)` + V `(3, 2)` |
| **Formula** | `Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V` |
| **Output** | `(3, 2)` per head — context-aware vector for each word |

---

### Step 11 — Concatenate Heads

| | |
|---|---|
| **Input** | 2 heads × `(3, 2)` |
| **Output** | `(3, 4)` — back to full `d_model` |

---

### Step 12 — Final Linear Projection (W_O)

| | |
|---|---|
| **Input** | `(3, 4)` |
| **Formula** | `output = concat(heads) · W_O` |
| **Output** | `(3, 4)` — final self-attention output |

---

### Step 13 — Add & Norm (Residual + LayerNorm)

**Input:** attention output `(3,4)` + original input from Step 4 `(3,4)`

**Formula:**

```
LayerNorm( x + Sublayer(x) )

where LayerNorm(z) = γ · (z - μ)/√(σ² + ε) + β
      μ = mean(z), σ² = variance(z)
```

**Output:** `(3, 4)`, mean ≈ 0, variance ≈ 1 per row, then rescaled by learnable `γ, β`

---

### Step 14 — Feed Forward NN

**Input:** `(3, 4)`

**Formula** (real model: 512 → 2048 → 512; toy: 4 → 16 → 4):

```
FFN(x) = max(0, x·W1 + b1)·W2 + b2      (ReLU activation)
```

**Output:** `(3, 4)`

---

### Step 15 — Add & Norm → Encoder Output

**Input:** FFN output `(3,4)` + Step 13 output `(3,4)`

**Formula:** same LayerNorm formula as Step 13

**Output (final encoder output, toy example):**

```
E_I    = [0.42, -0.18,  0.55,  0.10]
E_am   = [0.30,  0.25, -0.12,  0.40]
E_fine = [0.18,  0.33,  0.20, -0.05]
```

**Shape:** `(seq_len=3, d_model=4)` — this feeds Cross-Attention as **K and V** for every decoder layer 🔗

---

## Part 2 — Decoder Pipeline

### Step 16 — Target Input (shifted right)

| | |
|---|---|
| **Target sentence** | `"நான் நலம்"` → shifted: `[<SOS>, நான்]` (predicting one word at a time) |
| **Token IDs (example)** | `[<SOS>=1, நான்=23]` |
| **Shape** | `(tgt_len=2,)` |

---

### Step 17 — Output Embedding

| | |
|---|---|
| **Formula** | same lookup as Step 3: `x_i = E[token_id_i]` |
| **Output** | `(2, 4)` |

---

### Step 18 — + Positional Encoding

| | |
|---|---|
| **Formula** | identical sin/cos formula from Step 4 |
| **Output** | `(2, 4)` |

---

### Step 19 — Q, K, V Projection (decoder's own)

**Formula:** `Q = X·W_Q`, `K = X·W_K`, `V = X·W_V` (decoder's **own** weight matrices — separate from the encoder's)

**Output:** Q, K, V each `(2, 4)` → split heads → `(2 heads, 2, 2)`

---

### Step 20 — MatMul Q×Kᵀ, then Scale

**Formula:** `scores = (Q·Kᵀ) / √d_k`

**Output:** `(2, 2)` per head

---

### Step 21 — Masking (the extra step vs encoder) 🎭

**Formula:**

```
masked_scores[i][j] = scores[i][j]   if j ≤ i
                     = -∞             if j > i
```

**Example matrix (2×2, row = query pos, col = key pos):**

|  | `<SOS>` | `நான்` |
|---|---|---|
| **`<SOS>`** | 0.5 | −∞ |
| **`நான்`** | 0.3 | 0.6 |

**Output:** `(2, 2)`, upper-right triangle = `-∞` → prevents attending to future tokens

---

### Step 22 — Softmax

**Formula:** `softmax(zᵢ) = e^(zᵢ)/Σⱼe^(zⱼ)` (the `-∞` becomes exactly `0`)

**Example result:**

|  | `<SOS>` | `நான்` |
|---|---|---|
| **`<SOS>`** | 1.0 | 0.0 |
| **`நான்`** | 0.4 | 0.6 |

**Output:** `(2, 2)` — "நான்" attends 40% to `<SOS>`, 60% to itself; nothing attends to future words

---

### Step 23 — MatMul weights × V, Concat, Final Linear

**Formula:** same as Steps 10–12

**Output:** `(2, 4)` — masked self-attention output

---

### Step 24 — Add & Norm

**Formula:** `LayerNorm(x + Sublayer(x))`

**Output:** `(2, 4)` — this becomes the **Query** for Cross-Attention

---

### Step 25 — Cross-Attention 🔀

**Input:** Query from Step 24 `(2, 4)` **[decoder]** + Key, Value from Step 15 `(3, 4)` **[encoder output]**

**Formula:**

```
CrossAttn(Q,K,V) = softmax(Q·Kᵀ/√d_k)·V
```

- Q comes from decoder → shape `(2, d_k)`
- K, V come from encoder → shape `(3, d_k)`
- `Q·Kᵀ` → `(2, 3)` — each target word scores against all 3 source words

**Example:** "நான்" (meaning "I") gets the highest attention weight on the encoder's `E_I` vector.

**Output:** `(2, 4)` — decoder length preserved, content infused with source-sentence context

---

### Step 26 — Add & Norm

**Output:** `(2, 4)`

---

### Step 27 — Feed Forward NN

**Formula:** `FFN(x) = max(0, x·W1+b1)·W2+b2`

**Output:** `(2, 4)`

---

### Step 28 — Add & Norm → Final Decoder Output

**Output:** `(2, 4)`

---

### Step 29 — Linear Layer

**Formula:** `logits = x · W_vocab + b`, `W_vocab ∈ ℝ^(d_model × vocab_size)`

**Input:** `(2, 4)`

**Output:** `(2, vocab_size=6)` — e.g. for position 2 ("நான்" processed), scores per vocab word:

```
logits = [0.5, 1.2, 3.8, 0.1, 0.9, 2.0]
```

---

### Step 30 — Softmax

**Formula:** `softmax(zᵢ) = e^(zᵢ)/Σⱼe^(zⱼ)`

**Output (probabilities for that position):**

```
probs = [0.03, 0.07, 0.82, 0.01, 0.05, 0.02]
```

---

### Step 31 — Predicted Word (Final Output) 🎯

**Formula:** `predicted_token = argmax(probs)`

**Output:** index 2 → **"நலம்"** (highest probability = 0.82)

**Full decoder output so far:** `<SOS> → நான் → நலம்`

**Loop:** "நலம்" fed back as next decoder input → model predicts `<EOS>` next → sentence complete:

> ### ✅ **"நான் நலம்"**

---

## 📋 Quick Formula Reference Sheet

| Component | Formula |
|---|---|
| Positional Encoding | `PE(pos,2i)=sin(pos/10000^(2i/d_model))`, `PE(pos,2i+1)=cos(pos/10000^(2i/d_model))` |
| Q, K, V projection | `Q=XW_Q`, `K=XW_K`, `V=XW_V` |
| Attention score | `scores = QKᵀ` |
| Scaling | `scaled = scores/√d_k` |
| Masking (decoder only) | `masked[i][j] = -∞ if j>i else scores[i][j]` |
| Softmax | `softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ` |
| Full attention | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V` |
| Add & Norm | `LayerNorm(x + Sublayer(x))`, `LayerNorm(z)=γ(z-μ)/√(σ²+ε)+β` |
| Feed Forward | `FFN(x) = max(0, xW1+b1)W2+b2` |
| Final Linear | `logits = xW_vocab + b` |
| Prediction | `argmax(softmax(logits))` |

---

## 🗺️ Full Pipeline at a Glance

```
ENCODER                                   DECODER
────────                                   ────────
Input Sentence                            Target Input (shifted right)
   │                                          │
Tokenization (BPE)                        Output Embedding
   │                                          │
Input Embedding                           + Positional Encoding
   │                                          │
+ Positional Encoding                     Masked Self-Attention (Q,K,V from decoder)
   │                                          │  ├─ MatMul QKᵀ → Scale → Mask → Softmax → ×V
Q, K, V Projection                            │
   │                                       Add & Norm
Split into Heads                              │
   │                                       Cross-Attention  ◄──────── Encoder Output (K, V)
MatMul QKᵀ → Scale → Softmax → ×V             │  (Q from decoder, K/V from encoder)
   │                                          │
Concat Heads → Linear (W_O)               Add & Norm
   │                                          │
Add & Norm                                Feed Forward NN
   │                                          │
Feed Forward NN                           Add & Norm
   │                                          │
Add & Norm → ENCODER OUTPUT               Linear Layer (→ vocab_size)
   (feeds Cross-Attention as K, V)             │
                                            Softmax
                                               │
                                            argmax → Predicted Token
                                               │
                                     (loop: feed back as next decoder input)
```

---

<p align="center"><i>Reference: "Attention Is All You Need" — Vaswani et al., 2017</i></p>
