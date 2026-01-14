# 文字生成

## 概述

使用 `generate()` 方法透過語言模型生成文字。通過生成策略和參數控制輸出品質和風格。

## 基本生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize input
inputs = tokenizer("Once upon a time", return_tensors="pt")

# Generate
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## 生成策略

### 貪婪解碼（Greedy Decoding）

在每一步選擇最高機率的 token（確定性）：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False  # Greedy decoding (default)
)
```

**使用時機**：事實性文字、翻譯、需要確定性的場合。

### 採樣（Sampling）

從機率分佈中隨機採樣：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
```

**使用時機**：創意寫作、多樣化輸出、開放式生成。

### 束搜尋（Beam Search）

並行探索多個假設：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    early_stopping=True
)
```

**使用時機**：翻譯、摘要、品質至關重要的場合。

### 對比搜尋（Contrastive Search）

平衡品質和多樣性：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    penalty_alpha=0.6,
    top_k=4
)
```

**使用時機**：長文本生成、減少重複。

## 關鍵參數

### 長度控制

**max_new_tokens**：生成的最大 token 數
```python
max_new_tokens=100  # Generate up to 100 new tokens
```

**max_length**：最大總長度（輸入 + 輸出）
```python
max_length=512  # Total sequence length
```

**min_new_tokens**：生成的最小 token 數
```python
min_new_tokens=50  # Force at least 50 tokens
```

**min_length**：最小總長度
```python
min_length=100
```

### Temperature（溫度）

控制隨機性（僅在採樣時使用）：

```python
temperature=1.0   # Default, balanced
temperature=0.7   # More focused, less random
temperature=1.5   # More creative, more random
```

較低溫度 → 更確定性
較高溫度 → 更隨機

### Top-K 採樣

僅考慮最可能的前 K 個 token：

```python
do_sample=True
top_k=50  # Sample from top 50 tokens
```

**常見值**：40-100 用於平衡輸出，10-20 用於集中輸出。

### Top-P（Nucleus）採樣

考慮累積機率 >= P 的 token：

```python
do_sample=True
top_p=0.95  # Sample from smallest set with 95% cumulative probability
```

**常見值**：0.9-0.95 用於平衡，0.7-0.85 用於集中。

### 重複懲罰（Repetition Penalty）

減少重複：

```python
repetition_penalty=1.2  # Penalize repeated tokens
```

**值**：1.0 = 無懲罰，1.2-1.5 = 中等，2.0+ = 強懲罰。

### 束搜尋參數

**num_beams**：束的數量
```python
num_beams=5  # Keep 5 hypotheses
```

**early_stopping**：當 num_beams 個句子完成時停止
```python
early_stopping=True
```

**no_repeat_ngram_size**：防止 n-gram 重複
```python
no_repeat_ngram_size=3  # Don't repeat any 3-gram
```

### 輸出控制

**num_return_sequences**：生成多個輸出
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    num_return_sequences=3  # Return 3 different sequences
)
```

**pad_token_id**：指定填充 token
```python
pad_token_id=tokenizer.eos_token_id
```

**eos_token_id**：在特定 token 處停止生成
```python
eos_token_id=tokenizer.eos_token_id
```

## 進階功能

### 批次生成

為多個提示生成：

```python
prompts = ["Hello, my name is", "Once upon a time"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

outputs = model.generate(**inputs, max_new_tokens=50)

for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Prompt {i}: {text}\n")
```

### 串流生成

生成時串流輸出 token：

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

generation_kwargs = dict(
    inputs,
    streamer=streamer,
    max_new_tokens=100
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)

thread.join()
```

### 受限生成

強制特定 token 序列：

```python
# Force generation to start with specific tokens
force_words = ["Paris", "France"]
force_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in force_words]

outputs = model.generate(
    **inputs,
    force_words_ids=force_words_ids,
    num_beams=5
)
```

### 引導和控制

**防止不良詞彙：**
```python
bad_words = ["offensive", "inappropriate"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]

outputs = model.generate(
    **inputs,
    bad_words_ids=bad_words_ids
)
```

### 生成配置

儲存和重用生成參數：

```python
from transformers import GenerationConfig

# Create config
generation_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Save
generation_config.save_pretrained("./my_generation_config")

# Load and use
generation_config = GenerationConfig.from_pretrained("./my_generation_config")
outputs = model.generate(**inputs, generation_config=generation_config)
```

## 模型特定的生成

### 聊天模型

使用聊天模板：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 編碼器-解碼器模型

對於 T5、BART 等：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# T5 uses task prefixes
input_text = "translate English to French: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 最佳化

### 快取

啟用 KV 快取以加速生成：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    use_cache=True  # Default, faster generation
)
```

### 靜態快取

用於固定序列長度：

```python
from transformers import StaticCache

cache = StaticCache(model.config, max_batch_size=1, max_cache_len=1024, device="cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    past_key_values=cache
)
```

### 注意力實作

使用 Flash Attention 加速：

```python
model = AutoModelForCausalLM.from_pretrained(
    "model-id",
    attn_implementation="flash_attention_2"
)
```

## 生成方案

### 創意寫作

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)
```

### 事實性生成

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,  # Greedy
    repetition_penalty=1.1
)
```

### 多樣化輸出

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    num_return_sequences=5,
    temperature=1.5,
    do_sample=True
)
```

### 長文本生成

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    penalty_alpha=0.6,  # Contrastive search
    top_k=4,
    repetition_penalty=1.2
)
```

### 翻譯/摘要

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=3
)
```

## 常見問題

**輸出重複：**
- 增加 repetition_penalty（1.2-1.5）
- 使用 no_repeat_ngram_size（2-3）
- 嘗試對比搜尋
- 降低溫度

**品質不佳：**
- 使用束搜尋（num_beams=5）
- 降低溫度
- 調整 top_k/top_p

**太確定性：**
- 啟用採樣（do_sample=True）
- 增加溫度（0.7-1.0）
- 調整 top_k/top_p

**生成緩慢：**
- 減少批次大小
- 啟用 use_cache=True
- 使用 Flash Attention
- 減少 max_new_tokens

## 最佳實踐

1. **從預設開始**：然後根據輸出調整
2. **使用適當策略**：貪婪用於事實性，採樣用於創意
3. **設定 max_new_tokens**：避免不必要的長生成
4. **啟用快取**：用於更快的序列生成
5. **調整溫度**：採樣時最具影響力的參數
6. **謹慎使用束搜尋**：較慢但品質更高
7. **測試不同種子**：用於採樣時的可重現性
8. **監控記憶體**：大束使用大量記憶體
