## ⚡ Inference II: Quantization

**Quantization** reduces the numerical precision of model weights (and sometimes activations) to make inference **faster and cheaper**.

Instead of storing weights as:

```
float32 (32 bits)
```

we use:

```
float16 (16 bits)
int8  (8 bits)
int4  (4 bits)
```

---

# 🧠 Why Quantization Works

Neural networks don’t need ultra-precise numbers to function well.

Example:

```
0.123456789  →  0.1235
```

Tiny rounding error → almost no impact on output.

So we trade:

| Precision ↓ | Memory ↓ | Speed ↑ |
| ----------- | -------- | ------- |

---

# 📦 What Gets Quantized?

### 1️⃣ Weights (most common)

Huge memory savings.

### 2️⃣ Activations (optional)

More aggressive optimization.

### 3️⃣ KV-cache (advanced optimization)

Very helpful for long-context inference.

---

# 🔢 Memory Impact Example

Suppose a 7B parameter model.

### FP16:

```
7B × 2 bytes ≈ 14 GB
```

### INT8:

```
7B × 1 byte ≈ 7 GB
```

### INT4:

```
7B × 0.5 byte ≈ 3.5 GB
```

Massive savings.

---

# 🏗 How It Works (Conceptually)

We map floating-point values to integers:

### Step 1 — Find range

```
min_weight, max_weight
```

### Step 2 — Scale

```
float_value ≈ scale × int_value
```

Where:

```
scale = (max - min) / 255   (for int8)
```

So at runtime:

* Multiply integers
* Rescale back

Modern GPUs are very efficient at this.

---

# 🎯 Types of Quantization

### 🔹 Post-Training Quantization (PTQ)

* Quantize after training
* Fast, simple
* Slight quality drop possible

Common in deployment.

---

### 🔹 Quantization-Aware Training (QAT)

* Simulate quantization during training
* Better accuracy retention
* More complex

---

# 🔥 Popular Quantization Methods for LLMs

* **Dynamic quantization**
* **Static quantization**
* **GPTQ**
* **AWQ**
* **BitsAndBytes (LLM.int8, 4-bit)**

Each balances:

* Speed
* Memory
* Accuracy

---

# ⚖ Tradeoffs

| Benefit                     | Cost                 |
| --------------------------- | -------------------- |
| Lower memory                | Slight accuracy loss |
| Faster inference            | Possible instability |
| Larger models on small GPUs | Extra engineering    |

---

# 🧠 Simple Mental Model

Think of it like compressing a high-resolution image:

* 4K → 1080p
  Still clear, but much smaller.

---

# 🚀 Why It Matters for Inference

Quantization enables:

* Running 7B–13B models on consumer GPUs
* Lower cloud costs
* Higher batch sizes
* Faster token throughput

---

If you'd like, I can next explain:

* Quantization + KV-cache interaction
* Why 4-bit works surprisingly well
* Per-channel vs per-tensor quantization
* Quantization math in matrix multiplication (C#-friendly explanation)
