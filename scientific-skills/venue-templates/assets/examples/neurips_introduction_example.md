# NeurIPS/ICML 引言範例

此範例展示了具有編號貢獻和技術精確性的獨特 ML 會議引言結構。

---

## 完整引言範例

**論文主題**：高效長上下文 Transformers

---

### 第 1 段：問題動機

```
Large language models (LLMs) have demonstrated remarkable capabilities in
natural language understanding, code generation, and reasoning tasks [1, 2, 3].
These capabilities scale with both model size and context length—longer
contexts enable processing of entire documents, multi-turn conversations,
and complex reasoning chains that span many steps [4, 5]. However, the
standard Transformer attention mechanism [6] has O(N²) time and memory
complexity with respect to sequence length N, creating a fundamental
bottleneck for processing long sequences. For a context window of 100K
tokens, computing full attention requires 10 billion scalar operations
and 40 GB of memory for the attention matrix alone, making training and
inference prohibitively expensive on current hardware.
```

**關鍵特徵**：
- 說明為何這很重要（LLM 能力）
- 連結到擴展性（更長上下文 = 更好性能）
- 具體數字（O(N²)、100K tokens、100 億運算、40 GB）
- 引用建立可信度

---

### 第 2 段：現有方法的局限性

```
Prior work has addressed attention efficiency through three main approaches.
Sparse attention patterns [7, 8, 9] reduce complexity to O(N√N) or O(N log N)
by restricting attention to local windows, fixed stride patterns, or learned
sparse masks. Linear attention approximations [10, 11, 12] reformulate
attention using kernel feature maps that enable O(N) computation, but
sacrifice the ability to model arbitrary pairwise interactions. Low-rank
factorizations [13, 14] approximate the attention matrix as a product of
smaller matrices, achieving efficiency at the cost of expressivity. While
these methods reduce theoretical complexity, they introduce approximation
errors that compound in deep networks, often resulting in 2-5% accuracy
degradation on long-range modeling benchmarks [15]. Perhaps more importantly,
they fundamentally change the attention mechanism, making it difficult to
apply advances in standard attention (e.g., rotary positional embeddings,
grouped-query attention) to efficient variants.
```

**關鍵特徵**：
- 有組織的先前工作分類
- 說明每種方法的複雜度
- 清楚指出局限性
- 量化缺點（2-5% 性能下降）
- 確認更深層問題（與新進展不相容）

---

### 第 3 段：您的方法（高階）

```
We take a different approach: rather than approximating attention, we
accelerate exact attention by optimizing memory access patterns. Our key
observation is that on modern GPUs, attention is bottlenecked by memory
bandwidth, not compute. Reading and writing the N × N attention matrix to
and from GPU high-bandwidth memory (HBM) dominates runtime, while the GPU's
tensor cores remain underutilized. We propose LongFlash, an IO-aware exact
attention algorithm that computes attention block-by-block in fast on-chip
SRAM, never materializing the full attention matrix in HBM. By carefully
orchestrating the tiling pattern and fusing the softmax computation with
matrix multiplications, LongFlash reduces HBM accesses from O(N²) to
O(N²d/M) where d is the head dimension and M is the SRAM size, achieving
asymptotically optimal IO complexity.
```

**關鍵特徵**：
- 與先前工作清楚區分（「different approach」）
- 明確說明關鍵洞察
- 解釋技術機制
- 量化複雜度改進
- 介紹方法名稱

---

### 第 4 段：貢獻（關鍵）

```
Our contributions are as follows:

• We propose LongFlash, an IO-aware exact attention algorithm that achieves
  2-4× speedup over FlashAttention [16] and up to 9× over standard PyTorch
  attention on sequences from 1K to 128K tokens (Section 3).

• We provide theoretical analysis proving that LongFlash achieves optimal
  IO complexity of O(N²d/M) among all algorithms that compute exact
  attention, and analyze the regime where our algorithm provides maximum
  benefit (Section 3.3).

• We introduce sequence parallelism techniques that enable LongFlash to
  scale to sequences of 1M+ tokens across multiple GPUs with near-linear
  weak scaling efficiency (Section 4).

• We demonstrate that LongFlash enables training with 8× longer contexts
  on the same hardware: we train a 7B parameter model on 128K token
  contexts using the same memory that previously limited us to 16K tokens
  (Section 5).

• We release optimized CUDA kernels achieving 80% of theoretical peak
  FLOPS on A100 and H100 GPUs, along with PyTorch and JAX bindings, at
  [anonymous URL] (Section 6).
```

**關鍵特徵**：
- 編號/項目符號格式
- 每個貢獻都具體且量化
- 每個聲明都有章節參考
- 包含方法論和實證貢獻
- 提及程式碼釋出
- 每個項目都是獨立完整的（單獨閱讀也能理解）

---

## 替代開場段落

### 方法論文範例

```
Scalable optimization algorithms are fundamental to modern machine learning.
Stochastic gradient descent (SGD) and its variants [1, 2, 3] have enabled
training of models with billions of parameters on massive datasets. However,
these first-order methods exhibit slow convergence on ill-conditioned
problems, often requiring thousands of iterations to converge on tasks
where second-order methods would converge in tens of iterations [4, 5].
```

### 應用論文範例

```
Drug discovery is a costly and time-consuming process, with the average new
drug requiring 10-15 years and $2.6 billion to develop [1]. Machine learning
offers the potential to accelerate this process by predicting molecular
properties, identifying promising candidates, and optimizing lead compounds
computationally [2, 3]. Recent successes in protein structure prediction [4]
and molecular generation [5] have demonstrated that deep learning can
capture complex chemical patterns, raising hopes for ML-driven drug discovery.
```

### 理論論文範例

```
Understanding why deep neural networks generalize well despite having more
parameters than training examples remains one of the central puzzles of
modern machine learning [1, 2]. Classical statistical learning theory
predicts that such overparameterized models should overfit dramatically,
yet in practice, large networks trained with SGD achieve excellent test
accuracy [3]. This gap between theory and practice has motivated a rich
literature on implicit regularization [4], neural tangent kernels [5],
and feature learning [6], but a complete theoretical picture remains elusive.
```

---

## 貢獻項目模板

### 新方法

```
• We propose [Method Name], a novel [type of method] that [key innovation]
  achieving [performance improvement] over [baseline] on [benchmark].
```

### 理論分析

```
• We prove that [statement], providing the first [type of result] for
  [problem setting]. This resolves an open question from [prior work].
```

### 實證研究

```
• We conduct a comprehensive evaluation of [N] methods across [M] datasets,
  revealing that [key finding] and identifying [failure mode/best practice].
```

### 程式碼/資料釋出

```
• We release [resource name], a [description] containing [scale/scope],
  available at [URL]. This enables [future work/reproducibility].
```

---

## 常見錯誤避免

### 模糊的貢獻

❌ **不佳**：
```
• We propose a novel method for attention
• We show our method is better than baselines
• We provide theoretical analysis
```

✅ **良好**：
```
• We propose LongFlash, achieving 2-4× speedup over FlashAttention
• We prove LongFlash achieves optimal O(N²d/M) IO complexity
• We enable 8× longer context training on fixed hardware budget
```

### 缺少量化

❌ **不佳**：「Our method significantly outperforms prior work」
✅ **良好**：「Our method improves accuracy by 3.2% on GLUE and 4.1% on SuperGLUE」

### 重疊的項目

❌ **不佳**：
```
• We propose a new attention mechanism
• We introduce LongFlash attention
• Our novel attention approach...
```
（這三點說的是同一件事）

### 埋沒的貢獻

❌ **不佳**：貢獻項目在第 2 頁末尾
✅ **良好**：貢獻項目在第 1 頁末尾清楚可見

---

## 另請參閱

- `ml_conference_style.md` - 完整 ML 會議指南
- `venue_writing_styles.md` - 跨發表場域風格比較

