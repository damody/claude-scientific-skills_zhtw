# Tokenizers（分詞器）

## 概述

Tokenizers 將文字轉換為模型可以處理的數值表示（token）。它們處理特殊 token、填充（padding）、截斷（truncation）和注意力遮罩（attention masks）。

## 載入 Tokenizers

### AutoTokenizer

自動為模型載入正確的 tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

從本地路徑載入：
```python
tokenizer = AutoTokenizer.from_pretrained("./local/tokenizer/path")
```

## 基本分詞

### 編碼文字

```python
# Simple encoding
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

# With text tokenization
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
```

### 解碼 Token

```python
token_ids = [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
text = tokenizer.decode(token_ids)
print(text)  # "hello, how are you?"

# Skip special tokens
text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(text)  # "hello, how are you?"
```

## `__call__` 方法

主要的分詞介面：

```python
# Single text
inputs = tokenizer("Hello, how are you?")

# Returns dictionary with input_ids, attention_mask
print(inputs)
# {
#   'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]
# }
```

多個文字：
```python
texts = ["Hello", "How are you?"]
inputs = tokenizer(texts, padding=True, truncation=True)
```

## 關鍵參數

### 回傳張量

**return_tensors**：輸出格式（"pt"、"tf"、"np"）
```python
# PyTorch tensors
inputs = tokenizer("text", return_tensors="pt")

# TensorFlow tensors
inputs = tokenizer("text", return_tensors="tf")

# NumPy arrays
inputs = tokenizer("text", return_tensors="np")
```

### 填充

**padding**：將序列填充到相同長度
```python
# Pad to longest sequence in batch
inputs = tokenizer(texts, padding=True)

# Pad to specific length
inputs = tokenizer(texts, padding="max_length", max_length=128)

# No padding
inputs = tokenizer(texts, padding=False)
```

**pad_to_multiple_of**：填充到指定值的倍數
```python
inputs = tokenizer(texts, padding=True, pad_to_multiple_of=8)
```

### 截斷

**truncation**：限制序列長度
```python
# Truncate to max_length
inputs = tokenizer(text, truncation=True, max_length=512)

# Truncate first sequence in pairs
inputs = tokenizer(text1, text2, truncation="only_first")

# Truncate second sequence
inputs = tokenizer(text1, text2, truncation="only_second")

# Truncate longest first (default for pairs)
inputs = tokenizer(text1, text2, truncation="longest_first", max_length=512)
```

### 最大長度

**max_length**：最大序列長度
```python
inputs = tokenizer(text, max_length=512, truncation=True)
```

### 額外輸出

**return_attention_mask**：包含注意力遮罩（預設為 True）
```python
inputs = tokenizer(text, return_attention_mask=True)
```

**return_token_type_ids**：句子對的區段 ID
```python
inputs = tokenizer(text1, text2, return_token_type_ids=True)
```

**return_offsets_mapping**：字元位置映射（僅 Fast tokenizers）
```python
inputs = tokenizer(text, return_offsets_mapping=True)
```

**return_length**：包含序列長度
```python
inputs = tokenizer(texts, padding=True, return_length=True)
```

## 特殊 Token

### 預定義特殊 Token

存取特殊 token：
```python
print(tokenizer.cls_token)      # [CLS] or <s>
print(tokenizer.sep_token)      # [SEP] or </s>
print(tokenizer.pad_token)      # [PAD]
print(tokenizer.unk_token)      # [UNK]
print(tokenizer.mask_token)     # [MASK]
print(tokenizer.eos_token)      # End of sequence
print(tokenizer.bos_token)      # Beginning of sequence

# Get IDs
print(tokenizer.cls_token_id)
print(tokenizer.sep_token_id)
```

### 添加特殊 Token

手動控制：
```python
# Automatically add special tokens (default True)
inputs = tokenizer(text, add_special_tokens=True)

# Skip special tokens
inputs = tokenizer(text, add_special_tokens=False)
```

### 自訂特殊 Token

```python
special_tokens_dict = {
    "additional_special_tokens": ["<CUSTOM>", "<SPECIAL>"]
}

num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added} tokens")

# Resize model embeddings after adding tokens
model.resize_token_embeddings(len(tokenizer))
```

## 句子對

分詞文字對：

```python
text1 = "What is the capital of France?"
text2 = "Paris is the capital of France."

# Automatically handles separation
inputs = tokenizer(text1, text2, padding=True, truncation=True)

# Results in: [CLS] text1 [SEP] text2 [SEP]
```

## 批次編碼

處理多個文字：

```python
texts = ["First text", "Second text", "Third text"]

# Basic batch encoding
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Access individual encodings
for i in range(len(texts)):
    input_ids = batch["input_ids"][i]
    attention_mask = batch["attention_mask"][i]
```

## Fast Tokenizers

使用基於 Rust 的 tokenizers 以獲得速度：

```python
from transformers import AutoTokenizer

# Automatically loads Fast version if available
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Check if Fast
print(tokenizer.is_fast)  # True

# Force Fast tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# Force slow (Python) tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
```

### Fast Tokenizer 功能

**偏移映射**（字元位置）：
```python
inputs = tokenizer("Hello world", return_offsets_mapping=True)
print(inputs["offset_mapping"])
# [(0, 0), (0, 5), (6, 11), (0, 0)]  # [CLS], "Hello", "world", [SEP]
```

**Token 到詞的映射**：
```python
encoding = tokenizer("Hello world")
word_ids = encoding.word_ids()
print(word_ids)  # [None, 0, 1, None]  # [CLS]=None, "Hello"=0, "world"=1, [SEP]=None
```

## 儲存 Tokenizers

本地儲存：
```python
tokenizer.save_pretrained("./my_tokenizer")
```

推送到 Hub：
```python
tokenizer.push_to_hub("username/my-tokenizer")
```

## 進階用法

### 詞彙表

存取詞彙表：
```python
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)

# Get token for ID
token = tokenizer.convert_ids_to_tokens(100)

# Get ID for token
token_id = tokenizer.convert_tokens_to_ids("hello")
```

### 編碼細節

取得詳細的編碼資訊：

```python
encoding = tokenizer("Hello world", return_tensors="pt")

# Original methods still available
tokens = encoding.tokens()
word_ids = encoding.word_ids()
sequence_ids = encoding.sequence_ids()
```

### 自訂預處理

子類別化以實現自訂行為：

```python
class CustomTokenizer(AutoTokenizer):
    def __call__(self, text, **kwargs):
        # Custom preprocessing
        text = text.lower().strip()
        return super().__call__(text, **kwargs)
```

## 聊天模板

用於對話模型：

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]

# Apply chat template
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(text)

# Tokenize directly
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
```

## 常見模式

### 模式 1：簡單文字分類

```python
texts = ["I love this!", "I hate this!"]
labels = [1, 0]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Use with model
outputs = model(**inputs, labels=torch.tensor(labels))
```

### 模式 2：問答

```python
question = "What is the capital?"
context = "Paris is the capital of France."

inputs = tokenizer(
    question,
    context,
    padding=True,
    truncation=True,
    max_length=384,
    return_tensors="pt"
)
```

### 模式 3：文字生成

```python
prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id
)

# Decode
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 模式 4：資料集分詞

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Apply to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## 最佳實踐

1. **始終指定 return_tensors**：用於模型輸入
2. **使用 padding 和 truncation**：用於批次處理
3. **明確設定 max_length**：防止記憶體問題
4. **使用 Fast tokenizers**：可用時以獲得速度
5. **處理 pad_token**：生成時若為 None 則設為 eos_token
6. **添加特殊 token**：保持啟用（預設）除非有特定原因
7. **調整嵌入大小**：添加自訂 token 後
8. **使用 skip_special_tokens 解碼**：獲得更乾淨的輸出
9. **使用批次處理**：提高資料集處理效率
10. **將 tokenizer 與模型一起儲存**：確保相容性

## 常見問題

**填充 token 未設定：**
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**序列太長：**
```python
# Enable truncation
inputs = tokenizer(text, truncation=True, max_length=512)
```

**詞彙表不匹配：**
```python
# Always load tokenizer and model from same checkpoint
tokenizer = AutoTokenizer.from_pretrained("model-id")
model = AutoModel.from_pretrained("model-id")
```

**注意力遮罩問題：**
```python
# Ensure attention_mask is passed
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"]
)
```
