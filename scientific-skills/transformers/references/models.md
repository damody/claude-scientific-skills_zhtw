# 模型載入和管理

## 概述

transformers 函式庫提供靈活的模型載入，具有自動架構檢測、裝置管理和配置控制。

## 載入模型

### AutoModel 類別

使用 AutoModel 類別進行自動架構選擇：

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

# Base model (no task head)
model = AutoModel.from_pretrained("bert-base-uncased")

# Sequence classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Causal language modeling (GPT-style)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Masked language modeling (BERT-style)
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Sequence-to-sequence (T5-style)
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

### 常見 AutoModel 類別

**NLP 任務：**
- `AutoModelForSequenceClassification`：文字分類、情感分析
- `AutoModelForTokenClassification`：NER（命名實體識別）、POS 標註
- `AutoModelForQuestionAnswering`：抽取式問答
- `AutoModelForCausalLM`：文字生成（GPT、Llama）
- `AutoModelForMaskedLM`：遮罩語言模型（BERT）
- `AutoModelForSeq2SeqLM`：翻譯、摘要（T5、BART）

**視覺任務：**
- `AutoModelForImageClassification`：圖像分類
- `AutoModelForObjectDetection`：物件偵測
- `AutoModelForImageSegmentation`：圖像分割

**音訊任務：**
- `AutoModelForAudioClassification`：音訊分類
- `AutoModelForSpeechSeq2Seq`：語音識別

**多模態：**
- `AutoModelForVision2Seq`：圖像描述、VQA（視覺問答）

## 載入參數

### 基本參數

**pretrained_model_name_or_path**：模型識別碼或本地路徑
```python
model = AutoModel.from_pretrained("bert-base-uncased")  # From Hub
model = AutoModel.from_pretrained("./local/model/path")  # From disk
```

**num_labels**：分類的輸出標籤數量
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)
```

**cache_dir**：自訂快取位置
```python
model = AutoModel.from_pretrained("model-id", cache_dir="./my_cache")
```

### 裝置管理

**device_map**：大型模型的自動裝置分配
```python
# Automatically distribute across GPUs and CPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto"
)

# Sequential placement
model = AutoModelForCausalLM.from_pretrained(
    "model-id",
    device_map="sequential"
)

# Custom device map
device_map = {
    "transformer.layers.0": 0,      # GPU 0
    "transformer.layers.1": 1,      # GPU 1
    "transformer.layers.2": "cpu",  # CPU
}
model = AutoModel.from_pretrained("model-id", device_map=device_map)
```

手動裝置放置：
```python
import torch
model = AutoModel.from_pretrained("model-id")
model.to("cuda:0")  # Move to GPU 0
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

### 精度控制

**torch_dtype**：設定模型精度
```python
import torch

# Float16 (half precision)
model = AutoModel.from_pretrained("model-id", torch_dtype=torch.float16)

# BFloat16 (better range than float16)
model = AutoModel.from_pretrained("model-id", torch_dtype=torch.bfloat16)

# Auto (use original dtype)
model = AutoModel.from_pretrained("model-id", torch_dtype="auto")
```

### 注意力實作

**attn_implementation**：選擇注意力機制
```python
# Scaled Dot Product Attention (PyTorch 2.0+, fastest)
model = AutoModel.from_pretrained("model-id", attn_implementation="sdpa")

# Flash Attention 2 (requires flash-attn package)
model = AutoModel.from_pretrained("model-id", attn_implementation="flash_attention_2")

# Eager (default, most compatible)
model = AutoModel.from_pretrained("model-id", attn_implementation="eager")
```

### 記憶體最佳化

**low_cpu_mem_usage**：減少載入時的 CPU 記憶體
```python
model = AutoModelForCausalLM.from_pretrained(
    "large-model-id",
    low_cpu_mem_usage=True,
    device_map="auto"
)
```

**load_in_8bit**：8 位元量化（需要 bitsandbytes）
```python
model = AutoModelForCausalLM.from_pretrained(
    "model-id",
    load_in_8bit=True,
    device_map="auto"
)
```

**load_in_4bit**：4 位元量化
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "model-id",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## 模型配置

### 使用自訂配置載入

```python
from transformers import AutoConfig, AutoModel

# Load and modify config
config = AutoConfig.from_pretrained("bert-base-uncased")
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# Initialize model with custom config
model = AutoModel.from_pretrained("bert-base-uncased", config=config)
```

### 僅從配置初始化

```python
config = AutoConfig.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(config)  # Random weights
```

## 模型模式

### 訓練模式 vs 評估模式

模型預設載入為評估模式：

```python
model = AutoModel.from_pretrained("model-id")
print(model.training)  # False

# Switch to training mode
model.train()

# Switch back to evaluation mode
model.eval()
```

評估模式會停用 dropout 並使用批次正規化統計資訊。

## 儲存模型

### 本地儲存

```python
model.save_pretrained("./my_model")
```

這會建立：
- `config.json`：模型配置
- `pytorch_model.bin` 或 `model.safetensors`：模型權重

### 儲存到 Hugging Face Hub

```python
model.push_to_hub("username/model-name")

# With custom commit message
model.push_to_hub("username/model-name", commit_message="Update model")

# Private repository
model.push_to_hub("username/model-name", private=True)
```

## 模型檢查

### 參數計數

```python
# Total parameters
total_params = model.num_parameters()

# Trainable parameters only
trainable_params = model.num_parameters(only_trainable=True)

print(f"Total: {total_params:,}")
print(f"Trainable: {trainable_params:,}")
```

### 記憶體佔用

```python
memory_bytes = model.get_memory_footprint()
memory_mb = memory_bytes / 1024**2
print(f"Memory: {memory_mb:.2f} MB")
```

### 模型架構

```python
print(model)  # Print full architecture

# Access specific components
print(model.config)
print(model.base_model)
```

## 前向傳播

基本推論：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model-id")
model = AutoModelForSequenceClassification.from_pretrained("model-id")

inputs = tokenizer("Sample text", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
predictions = logits.argmax(dim=-1)
```

## 模型格式

### SafeTensors vs PyTorch

SafeTensors 更快且更安全：

```python
# Save as safetensors (recommended)
model.save_pretrained("./model", safe_serialization=True)

# Load either format automatically
model = AutoModel.from_pretrained("./model")
```

### ONNX 匯出

匯出以進行最佳化推論：

```python
from transformers.onnx import export

# Export to ONNX
export(
    tokenizer=tokenizer,
    model=model,
    config=config,
    output=Path("model.onnx")
)
```

## 最佳實踐

1. **使用 AutoModel 類別**：自動架構檢測
2. **明確指定 dtype**：控制精度和記憶體
3. **使用 device_map="auto"**：用於大型模型
4. **啟用 low_cpu_mem_usage**：載入大型模型時
5. **使用 safetensors 格式**：更快且更安全的序列化
6. **檢查 model.training**：確保任務的正確模式
7. **考慮量化**：用於資源受限裝置上的部署
8. **本地快取模型**：設定 TRANSFORMERS_CACHE 環境變數

## 常見問題

**CUDA 記憶體不足：**
```python
# Use smaller precision
model = AutoModel.from_pretrained("model-id", torch_dtype=torch.float16)

# Or use quantization
model = AutoModel.from_pretrained("model-id", load_in_8bit=True)

# Or use CPU
model = AutoModel.from_pretrained("model-id", device_map="cpu")
```

**載入緩慢：**
```python
# Enable low CPU memory mode
model = AutoModel.from_pretrained("model-id", low_cpu_mem_usage=True)
```

**找不到模型：**
```python
# Verify model ID on hub.co
# Check authentication for private models
from huggingface_hub import login
login()
```
