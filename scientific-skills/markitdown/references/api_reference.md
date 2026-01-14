# MarkItDown API 參考

## 核心類別

### MarkItDown

將檔案轉換為 Markdown 的主要類別。

```python
from markitdown import MarkItDown

md = MarkItDown(
    llm_client=None,
    llm_model=None,
    llm_prompt=None,
    docintel_endpoint=None,
    enable_plugins=False
)
```

#### 參數

| 參數 | 類型 | 預設值 | 描述 |
|-----------|------|---------|-------------|
| `llm_client` | OpenAI client | `None` | 用於 AI 圖片描述的 OpenAI 相容客戶端 |
| `llm_model` | str | `None` | 用於圖片描述的模型名稱（例如 "anthropic/claude-sonnet-4.5"） |
| `llm_prompt` | str | `None` | 圖片描述的自訂提示 |
| `docintel_endpoint` | str | `None` | Azure 文件智慧端點 |
| `enable_plugins` | bool | `False` | 啟用第三方外掛 |

#### 方法

##### convert()

將檔案轉換為 Markdown。

```python
result = md.convert(
    source,
    file_extension=None
)
```

**參數**：
- `source` (str)：要轉換的檔案路徑
- `file_extension` (str, 可選)：覆蓋檔案副檔名檢測

**返回值**：`DocumentConverterResult` 物件

**範例**：
```python
result = md.convert("document.pdf")
print(result.text_content)
```

##### convert_stream()

從類檔案二進位串流轉換。

```python
result = md.convert_stream(
    stream,
    file_extension
)
```

**參數**：
- `stream` (BinaryIO)：二進位類檔案物件（例如以 `"rb"` 模式開啟的檔案）
- `file_extension` (str)：用於決定轉換方法的檔案副檔名（例如 ".pdf"）

**返回值**：`DocumentConverterResult` 物件

**範例**：
```python
with open("document.pdf", "rb") as f:
    result = md.convert_stream(f, file_extension=".pdf")
    print(result.text_content)
```

**重要**：串流必須以二進位模式（`"rb"`）開啟，而非文字模式。

## 結果物件

### DocumentConverterResult

轉換操作的結果。

#### 屬性

| 屬性 | 類型 | 描述 |
|-----------|------|-------------|
| `text_content` | str | 轉換後的 Markdown 文字 |
| `title` | str | 文件標題（如果可用） |

#### 範例

```python
result = md.convert("paper.pdf")

# 存取內容
content = result.text_content

# 存取標題（如果可用）
title = result.title
```

## 自訂轉換器

您可以透過實作 `DocumentConverter` 介面來建立自訂文件轉換器。

### DocumentConverter 介面

```python
from markitdown import DocumentConverter

class CustomConverter(DocumentConverter):
    def convert(self, stream, file_extension):
        """
        從二進位串流轉換文件。

        參數：
            stream (BinaryIO)：二進位類檔案物件
            file_extension (str)：檔案副檔名（例如 ".custom"）

        返回值：
            DocumentConverterResult：轉換結果
        """
        # 您的轉換邏輯在此
        pass
```

### 註冊自訂轉換器

```python
from markitdown import MarkItDown, DocumentConverter, DocumentConverterResult

class MyCustomConverter(DocumentConverter):
    def convert(self, stream, file_extension):
        content = stream.read().decode('utf-8')
        markdown_text = f"# Custom Format\n\n{content}"
        return DocumentConverterResult(
            text_content=markdown_text,
            title="Custom Document"
        )

# 建立 MarkItDown 實例
md = MarkItDown()

# 為 .custom 檔案註冊自訂轉換器
md.register_converter(".custom", MyCustomConverter())

# 使用它
result = md.convert("myfile.custom")
```

## 外掛系統

### 尋找外掛

在 GitHub 上搜尋 `#markitdown-plugin` 標籤。

### 使用外掛

```python
from markitdown import MarkItDown

# 啟用外掛
md = MarkItDown(enable_plugins=True)
result = md.convert("document.pdf")
```

### 建立外掛

外掛是向 MarkItDown 註冊轉換器的 Python 套件。

**外掛結構**：
```
my-markitdown-plugin/
├── setup.py
├── my_plugin/
│   ├── __init__.py
│   └── converter.py
└── README.md
```

**setup.py**：
```python
from setuptools import setup

setup(
    name="markitdown-my-plugin",
    version="0.1.0",
    packages=["my_plugin"],
    entry_points={
        "markitdown.plugins": [
            "my_plugin = my_plugin.converter:MyConverter",
        ],
    },
)
```

**converter.py**：
```python
from markitdown import DocumentConverter, DocumentConverterResult

class MyConverter(DocumentConverter):
    def convert(self, stream, file_extension):
        # 您的轉換邏輯
        content = stream.read()
        markdown = self.process(content)
        return DocumentConverterResult(
            text_content=markdown,
            title="My Document"
        )

    def process(self, content):
        # 處理內容
        return "# Converted Content\n\n..."
```

## AI 增強轉換

### 使用 OpenRouter 進行圖片描述

```python
from markitdown import MarkItDown
from openai import OpenAI

# 初始化 OpenRouter 客戶端（OpenAI 相容 API）
client = OpenAI(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1"
)

# 建立支援 AI 的 MarkItDown
md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",  # 推薦用於科學視覺
    llm_prompt="Describe this image in detail for scientific documentation"
)

# 轉換含圖片的檔案
result = md.convert("presentation.pptx")
```

### 透過 OpenRouter 可用的模型

支援視覺的熱門模型：
- `anthropic/claude-sonnet-4.5` - **推薦用於科學視覺**
- `anthropic/claude-opus-4.5` - 進階視覺模型
- `openai/gpt-4o` - GPT-4 Omni
- `openai/gpt-4-vision` - GPT-4 Vision
- `google/gemini-pro-vision` - Gemini Pro Vision

完整清單請見 https://openrouter.ai/models。

### 自訂提示

```python
# 用於科學圖表
scientific_prompt = """
Analyze this scientific diagram or chart. Describe:
1. The type of visualization (graph, chart, diagram, etc.)
2. Key data points or trends
3. Labels and axes
4. Scientific significance
Be precise and technical.
"""

md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",
    llm_prompt=scientific_prompt
)
```

## Azure 文件智慧

### 設定

1. 建立 Azure 文件智慧資源
2. 取得端點 URL
3. 設定驗證

### 使用方式

```python
from markitdown import MarkItDown

md = MarkItDown(
    docintel_endpoint="https://YOUR-RESOURCE.cognitiveservices.azure.com/"
)

result = md.convert("complex_document.pdf")
```

### 驗證

設定環境變數：
```bash
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-key"
```

或以程式方式傳遞憑證。

## 錯誤處理

```python
from markitdown import MarkItDown

md = MarkItDown()

try:
    result = md.convert("document.pdf")
    print(result.text_content)
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid file format: {e}")
except Exception as e:
    print(f"Conversion error: {e}")
```

## 效能技巧

### 1. 重複使用 MarkItDown 實例

```python
# 良好做法：建立一次，多次使用
md = MarkItDown()

for file in files:
    result = md.convert(file)
    process(result)
```

### 2. 對大型檔案使用串流

```python
# 對於大型檔案
with open("large_file.pdf", "rb") as f:
    result = md.convert_stream(f, file_extension=".pdf")
```

### 3. 批次處理

```python
from concurrent.futures import ThreadPoolExecutor

md = MarkItDown()

def convert_file(filepath):
    return md.convert(filepath)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(convert_file, file_list)
```

## 重大變更（v0.0.1 到 v0.1.0）

1. **依賴項**：現在組織為可選功能群組
   ```bash
   # 舊版
   pip install markitdown

   # 新版
   pip install 'markitdown[all]'
   ```

2. **convert_stream()**：現在需要二進位類檔案物件
   ```python
   # 舊版（也接受文字）
   with open("file.pdf", "r") as f:  # 文字模式
       result = md.convert_stream(f)

   # 新版（僅限二進位）
   with open("file.pdf", "rb") as f:  # 二進位模式
       result = md.convert_stream(f, file_extension=".pdf")
   ```

3. **DocumentConverter 介面**：改為從串流讀取而非檔案路徑
   - 不建立暫存檔案
   - 更高效的記憶體使用
   - 外掛需要更新

## 版本相容性

- **Python**：需要 3.10 或更高版本
- **依賴項**：查看 `setup.py` 了解版本限制
- **OpenAI**：與 OpenAI Python SDK v1.0+ 相容

## 環境變數

| 變數 | 描述 | 範例 |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | 用於圖片描述的 OpenRouter API 金鑰 | `sk-or-v1-...` |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure DI 驗證 | `key123...` |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure DI 端點 | `https://...` |

