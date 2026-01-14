# 訓練和微調

## 概述

使用 Trainer API 在自訂資料集上微調預訓練模型。Trainer 處理訓練迴圈、梯度累積、混合精度、日誌記錄和檢查點儲存。

## 基本微調工作流程

### 步驟 1：載入和預處理資料

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("yelp_review_full")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Tokenize
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
```

### 步驟 2：載入模型

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5  # Number of classes
)
```

### 步驟 3：定義指標

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

### 步驟 4：配置訓練

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```

### 步驟 5：建立 Trainer 並訓練

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)
```

### 步驟 6：儲存模型

```python
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Or push to Hub
trainer.push_to_hub("username/my-finetuned-model")
```

## TrainingArguments 參數

### 基本參數

**output_dir**：檢查點和日誌的目錄
```python
output_dir="./results"
```

**num_train_epochs**：訓練週期數
```python
num_train_epochs=3
```

**per_device_train_batch_size**：每個 GPU/CPU 的批次大小
```python
per_device_train_batch_size=8
```

**learning_rate**：最佳化器學習率
```python
learning_rate=2e-5  # Common for BERT-style models
learning_rate=5e-5  # Common for smaller models
```

**weight_decay**：L2 正則化
```python
weight_decay=0.01
```

### 評估和儲存

**eval_strategy**：何時評估（"no"、"steps"、"epoch"）
```python
eval_strategy="epoch"  # Evaluate after each epoch
eval_strategy="steps"  # Evaluate every eval_steps
```

**save_strategy**：何時儲存檢查點
```python
save_strategy="epoch"
save_strategy="steps"
save_steps=500
```

**load_best_model_at_end**：訓練後載入最佳檢查點
```python
load_best_model_at_end=True
metric_for_best_model="accuracy"  # Metric to compare
```

### 最佳化

**gradient_accumulation_steps**：跨多步累積梯度
```python
gradient_accumulation_steps=4  # Effective batch size = batch_size * 4
```

**fp16**：啟用混合精度（NVIDIA GPU）
```python
fp16=True
```

**bf16**：啟用 bfloat16（較新 GPU）
```python
bf16=True
```

**gradient_checkpointing**：以計算換取記憶體
```python
gradient_checkpointing=True  # Slower but uses less memory
```

**optim**：最佳化器選擇
```python
optim="adamw_torch"  # Default
optim="adamw_8bit"    # 8-bit Adam (requires bitsandbytes)
optim="adafactor"     # Memory-efficient alternative
```

### 學習率排程

**lr_scheduler_type**：學習率排程
```python
lr_scheduler_type="linear"       # Linear decay
lr_scheduler_type="cosine"       # Cosine annealing
lr_scheduler_type="constant"     # No decay
lr_scheduler_type="constant_with_warmup"
```

**warmup_steps** 或 **warmup_ratio**：預熱期間
```python
warmup_steps=500
# Or
warmup_ratio=0.1  # 10% of total steps
```

### 日誌記錄

**logging_dir**：TensorBoard 日誌目錄
```python
logging_dir="./logs"
```

**logging_steps**：每 N 步記錄
```python
logging_steps=10
```

**report_to**：日誌整合
```python
report_to=["tensorboard"]
report_to=["wandb"]
report_to=["tensorboard", "wandb"]
```

### 分散式訓練

**ddp_backend**：分散式後端
```python
ddp_backend="nccl"  # For multi-GPU
```

**deepspeed**：DeepSpeed 配置檔案
```python
deepspeed="ds_config.json"
```

## 資料整理器（Data Collators）

處理動態填充和特殊預處理：

### DataCollatorWithPadding

將序列填充到批次中最長的：
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

### DataCollatorForLanguageModeling

用於遮罩語言模型：
```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```

### DataCollatorForSeq2Seq

用於序列到序列任務：
```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)
```

## 自訂訓練

### 自訂 Trainer

覆寫方法以實現自訂行為：

```python
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Custom loss computation
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
```

### 自訂回呼

監控和控制訓練：

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed")
        # Custom logic here
        return control

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[CustomCallback],
)
```

## 進階訓練技術

### 參數高效微調（PEFT）

使用 LoRA 進行高效微調：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows reduced parameter count

# Train normally with Trainer
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### 梯度檢查點

以速度換取記憶體：

```python
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    gradient_checkpointing=True,
    ...
)
```

### 混合精度訓練

```python
training_args = TrainingArguments(
    fp16=True,  # For NVIDIA GPUs with Tensor Cores
    # or
    bf16=True,  # For newer GPUs (A100, H100)
    ...
)
```

### DeepSpeed 整合

用於非常大的模型：

```python
# ds_config.json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

```python
training_args = TrainingArguments(
    deepspeed="ds_config.json",
    ...
)
```

## 訓練技巧

### 超參數調整

常見起點：
- **學習率**：BERT 類模型 2e-5 至 5e-5，較小模型 1e-4 至 1e-3
- **批次大小**：根據 GPU 記憶體 8-32
- **週期數**：微調 2-4，領域適應更多
- **預熱**：總步數的 10%

使用 Optuna 進行超參數搜尋：

```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5
    )

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
    }

trainer = Trainer(model_init=model_init, args=training_args, ...)
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
)
```

### 監控訓練

使用 TensorBoard：
```bash
tensorboard --logdir ./logs
```

或 Weights & Biases：
```python
import wandb
wandb.init(project="my-project")

training_args = TrainingArguments(
    report_to=["wandb"],
    ...
)
```

### 恢復訓練

從檢查點恢復：
```python
trainer.train(resume_from_checkpoint="./results/checkpoint-1000")
```

## 常見問題

**CUDA 記憶體不足：**
- 減少批次大小
- 啟用梯度檢查點
- 使用梯度累積
- 使用 8 位元最佳化器

**過擬合：**
- 增加 weight_decay
- 添加 dropout
- 使用早停
- 減少模型大小或訓練週期數

**訓練緩慢：**
- 增加批次大小
- 啟用混合精度（fp16/bf16）
- 使用多 GPU
- 最佳化資料載入

## 最佳實踐

1. **從小處開始**：先在小型資料集子集上測試
2. **使用評估**：監控驗證指標
3. **儲存檢查點**：啟用 save_strategy
4. **廣泛記錄**：使用 TensorBoard 或 W&B
5. **嘗試不同學習率**：從 2e-5 開始
6. **使用預熱**：有助於訓練穩定性
7. **啟用混合精度**：更快的訓練
8. **考慮 PEFT**：用於資源有限的大型模型
