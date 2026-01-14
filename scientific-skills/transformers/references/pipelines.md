# Pipeline API 參考

## 概述

Pipelines 提供使用預訓練模型進行推論的最簡單方式。它們抽象化分詞、模型載入和後處理，為數十種任務提供統一介面。

## 基本用法

透過指定任務建立 pipeline：

```python
from transformers import pipeline

# Auto-select default model for task
pipe = pipeline("text-classification")
result = pipe("This is great!")
```

或指定模型：

```python
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
```

## 支援的任務

### 自然語言處理

**text-generation**：生成文字延續
```python
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=50, num_return_sequences=2)
```

**text-classification**：將文字分類到類別
```python
classifier = pipeline("text-classification")
result = classifier("I love this product!")  # Returns label and score
```

**token-classification**：標記個別 token（NER、POS 標註）
```python
ner = pipeline("token-classification", model="dslim/bert-base-NER")
entities = ner("Hugging Face is based in New York City")
```

**question-answering**：從上下文抽取答案
```python
qa = pipeline("question-answering")
result = qa(question="What is the capital?", context="Paris is the capital of France.")
```

**fill-mask**：預測遮罩 token
```python
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("Paris is the [MASK] of France")
```

**summarization**：摘要長文字
```python
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer("Long article text...", max_length=130, min_length=30)
```

**translation**：語言間翻譯
```python
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")
```

**zero-shot-classification**：無需訓練資料即可分類
```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This is a course about Python programming",
    candidate_labels=["education", "politics", "business"]
)
```

**sentiment-analysis**：專注於情感的 text-classification 別名
```python
sentiment = pipeline("sentiment-analysis")
result = sentiment("This product exceeded my expectations!")
```

### 電腦視覺

**image-classification**：分類圖像
```python
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("path/to/image.jpg")
# Or use PIL Image or URL
from PIL import Image
result = classifier(Image.open("image.jpg"))
```

**object-detection**：偵測圖像中的物件
```python
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
results = detector("image.jpg")  # Returns bounding boxes and labels
```

**image-segmentation**：分割圖像
```python
segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
segments = segmenter("image.jpg")
```

**depth-estimation**：從圖像估計深度
```python
depth = pipeline("depth-estimation", model="Intel/dpt-large")
result = depth("image.jpg")
```

**zero-shot-image-classification**：無需訓練即可分類圖像
```python
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
result = classifier("image.jpg", candidate_labels=["cat", "dog", "bird"])
```

### 音訊

**automatic-speech-recognition**：轉錄語音
```python
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
text = asr("audio.mp3")
```

**audio-classification**：分類音訊
```python
classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
result = classifier("audio.wav")
```

**text-to-speech**：從文字生成語音（使用特定模型）
```python
tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")
audio = tts("Hello, this is a test")
```

### 多模態

**visual-question-answering**：回答關於圖像的問題
```python
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
result = vqa(image="image.jpg", question="What color is the car?")
```

**document-question-answering**：回答關於文件的問題
```python
doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
result = doc_qa(image="document.png", question="What is the invoice number?")
```

**image-to-text**：為圖像生成描述
```python
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
caption = captioner("image.jpg")
```

## Pipeline 參數

### 常見參數

**model**：模型識別碼或路徑
```python
pipe = pipeline("task", model="model-id")
```

**device**：GPU 裝置索引（-1 為 CPU，0+ 為 GPU）
```python
pipe = pipeline("task", device=0)  # Use first GPU
```

**device_map**：大型模型的自動裝置分配
```python
pipe = pipeline("task", model="large-model", device_map="auto")
```

**dtype**：模型精度（減少記憶體）
```python
import torch
pipe = pipeline("task", torch_dtype=torch.float16)
```

**batch_size**：一次處理多個輸入
```python
pipe = pipeline("task", batch_size=8)
results = pipe(["text1", "text2", "text3"])
```

**framework**：選擇 PyTorch 或 TensorFlow
```python
pipe = pipeline("task", framework="pt")  # or "tf"
```

## 批次處理

高效處理多個輸入：

```python
classifier = pipeline("text-classification")
texts = ["Great product!", "Terrible experience", "Just okay"]
results = classifier(texts)
```

對於大型資料集，使用生成器或 KeyDataset：

```python
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("dataset-name", split="test")
pipe = pipeline("task", device=0)

for output in pipe(KeyDataset(dataset, "text")):
    print(output)
```

## 效能最佳化

### GPU 加速

始終指定裝置以使用 GPU：
```python
pipe = pipeline("task", device=0)
```

### 混合精度

使用 float16 在支援的 GPU 上獲得 2 倍加速：
```python
import torch
pipe = pipeline("task", torch_dtype=torch.float16, device=0)
```

### 批次處理指南

- **CPU**：通常跳過批次處理
- **長度可變的 GPU**：可能降低效率
- **長度相近的 GPU**：顯著加速
- **即時應用**：跳過批次處理（增加延遲）

```python
# Good for throughput
pipe = pipeline("task", batch_size=32, device=0)
results = pipe(list_of_texts)
```

### 串流輸出

對於文字生成，生成時串流 token：

```python
from transformers import TextStreamer

generator = pipeline("text-generation", model="gpt2", streamer=TextStreamer())
generator("The future of AI", max_length=100)
```

## 自訂 Pipeline 配置

分別指定 tokenizer 和模型：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("model-id")
model = AutoModelForSequenceClassification.from_pretrained("model-id")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
```

使用自訂 pipeline 類別：

```python
from transformers import TextClassificationPipeline

class CustomPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs, **kwargs):
        # Custom post-processing
        return super().postprocess(model_outputs, **kwargs)

pipe = pipeline("text-classification", model="model-id", pipeline_class=CustomPipeline)
```

## 輸入格式

Pipelines 接受各種輸入類型：

**文字任務**：字串或字串列表
```python
pipe("single text")
pipe(["text1", "text2"])
```

**圖像任務**：URL、檔案路徑、PIL 圖像或 numpy 陣列
```python
pipe("https://example.com/image.jpg")
pipe("local/path/image.png")
pipe(PIL.Image.open("image.jpg"))
pipe(numpy_array)
```

**音訊任務**：檔案路徑、numpy 陣列或原始波形
```python
pipe("audio.mp3")
pipe(audio_array)
```

## 錯誤處理

處理常見問題：

```python
try:
    result = pipe(input_data)
except Exception as e:
    if "CUDA out of memory" in str(e):
        # Reduce batch size or use CPU
        pipe = pipeline("task", device=-1)
    elif "does not appear to have a file named" in str(e):
        # Model not found
        print("Check model identifier")
    else:
        raise
```

## 最佳實踐

1. **使用 pipelines 進行原型設計**：無需樣板程式碼即可快速迭代
2. **明確指定模型**：預設模型可能會變更
3. **可用時啟用 GPU**：顯著加速
4. **使用批次處理提高吞吐量**：處理多個輸入時
5. **考慮記憶體用量**：對大批次使用 float16 或較小模型
6. **本地快取模型**：避免重複下載
