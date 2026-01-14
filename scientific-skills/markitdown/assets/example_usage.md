# MarkItDown 使用範例

本文件提供在各種情境中使用 MarkItDown 的實用範例。

## 基本範例

### 1. 簡單檔案轉換

```python
from markitdown import MarkItDown

md = MarkItDown()

# 轉換 PDF
result = md.convert("research_paper.pdf")
print(result.text_content)

# 轉換 Word 文件
result = md.convert("manuscript.docx")
print(result.text_content)

# 轉換 PowerPoint
result = md.convert("presentation.pptx")
print(result.text_content)
```

### 2. 儲存到檔案

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")

with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.text_content)
```

### 3. 從串流轉換

```python
from markitdown import MarkItDown

md = MarkItDown()

with open("document.pdf", "rb") as f:
    result = md.convert_stream(f, file_extension=".pdf")
    print(result.text_content)
```

## 科學工作流程

### 轉換研究論文

```python
from markitdown import MarkItDown
from pathlib import Path

md = MarkItDown()

# 轉換目錄中的所有論文
papers_dir = Path("research_papers/")
output_dir = Path("markdown_papers/")
output_dir.mkdir(exist_ok=True)

for paper in papers_dir.glob("*.pdf"):
    result = md.convert(str(paper))

    # 使用原始檔名儲存
    output_file = output_dir / f"{paper.stem}.md"
    output_file.write_text(result.text_content)

    print(f"Converted: {paper.name}")
```

### 從 Excel 擷取表格

```python
from markitdown import MarkItDown

md = MarkItDown()

# 將 Excel 轉換為 Markdown 表格
result = md.convert("experimental_data.xlsx")

# 結果包含 Markdown 格式的表格
print(result.text_content)

# 儲存以供進一步處理
with open("data_tables.md", "w") as f:
    f.write(result.text_content)
```

### 處理簡報投影片

```python
from markitdown import MarkItDown
from openai import OpenAI

# 使用 AI 描述圖片
client = OpenAI()
md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",
    llm_prompt="Describe this scientific slide, focusing on data and key findings"
)

result = md.convert("conference_talk.pptx")

# 儲存並附加元資料
output = f"""# Conference Talk

{result.text_content}
"""

with open("talk_notes.md", "w") as f:
    f.write(output)
```

## AI 增強轉換

### 詳細圖片描述

```python
from markitdown import MarkItDown
from openai import OpenAI

# 初始化 OpenRouter 客戶端
client = OpenAI(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1"
)

# 科學圖表分析
scientific_prompt = """
Analyze this scientific figure. Describe:
- Type of visualization (graph, microscopy, diagram, etc.)
- Key data points and trends
- Axes, labels, and legends
- Scientific significance
Be technical and precise.
"""

md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",  # 推薦用於科學視覺
    llm_prompt=scientific_prompt
)

# 轉換含圖表的論文
result = md.convert("paper_with_figures.pdf")
print(result.text_content)
```

### 針對不同檔案使用不同提示

```python
from markitdown import MarkItDown
from openai import OpenAI

# 初始化 OpenRouter 客戶端
client = OpenAI(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1"
)

# 科學論文 - 使用 Claude 進行技術分析
scientific_md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",
    llm_prompt="Describe scientific figures with technical precision"
)

# 簡報 - 使用 GPT-4o 進行視覺理解
presentation_md = MarkItDown(
    llm_client=client,
    llm_model="anthropic/claude-sonnet-4.5",
    llm_prompt="Summarize slide content and key visual elements"
)

# 針對每個檔案使用適當的實例
paper_result = scientific_md.convert("research.pdf")
slides_result = presentation_md.convert("talk.pptx")
```

## 批次處理

### 處理多個檔案

```python
from markitdown import MarkItDown
from pathlib import Path

md = MarkItDown()

files_to_convert = [
    "paper1.pdf",
    "data.xlsx",
    "presentation.pptx",
    "notes.docx"
]

for file in files_to_convert:
    try:
        result = md.convert(file)
        output = Path(file).stem + ".md"

        with open(output, "w") as f:
            f.write(result.text_content)

        print(f"✓ {file} -> {output}")
    except Exception as e:
        print(f"✗ Error converting {file}: {e}")
```

### 平行處理

```python
from markitdown import MarkItDown
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def convert_file(filepath):
    md = MarkItDown()
    result = md.convert(filepath)

    output = Path(filepath).stem + ".md"
    with open(output, "w") as f:
        f.write(result.text_content)

    return filepath, output

files = list(Path("documents/").glob("*.pdf"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(convert_file, [str(f) for f in files])

    for input_file, output_file in results:
        print(f"Converted: {input_file} -> {output_file}")
```

## 整合範例

### 文獻審閱流程

```python
from markitdown import MarkItDown
from pathlib import Path
import json

md = MarkItDown()

# 轉換論文並建立元資料
papers_dir = Path("literature/")
output_dir = Path("literature_markdown/")
output_dir.mkdir(exist_ok=True)

catalog = []

for paper in papers_dir.glob("*.pdf"):
    result = md.convert(str(paper))

    # 儲存 Markdown
    md_file = output_dir / f"{paper.stem}.md"
    md_file.write_text(result.text_content)

    # 儲存元資料
    catalog.append({
        "title": result.title or paper.stem,
        "source": paper.name,
        "markdown": str(md_file),
        "word_count": len(result.text_content.split())
    })

# 儲存目錄
with open(output_dir / "catalog.json", "w") as f:
    json.dump(catalog, f, indent=2)
```

### 資料擷取流程

```python
from markitdown import MarkItDown
import re

md = MarkItDown()

# 將 Excel 資料轉換為 Markdown
result = md.convert("experimental_results.xlsx")

# 擷取表格（Markdown 表格以 | 開頭）
tables = []
current_table = []
in_table = False

for line in result.text_content.split('\n'):
    if line.strip().startswith('|'):
        in_table = True
        current_table.append(line)
    elif in_table:
        if current_table:
            tables.append('\n'.join(current_table))
            current_table = []
        in_table = False

# 處理每個表格
for i, table in enumerate(tables):
    print(f"Table {i+1}:")
    print(table)
    print("\n" + "="*50 + "\n")
```

### YouTube 影片文字稿分析

```python
from markitdown import MarkItDown

md = MarkItDown()

# 取得文字稿
video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
result = md.convert(video_url)

# 儲存文字稿
with open("lecture_transcript.md", "w") as f:
    f.write(f"# Lecture Transcript\n\n")
    f.write(f"**Source**: {video_url}\n\n")
    f.write(result.text_content)
```

## 錯誤處理

### 穩健的轉換

```python
from markitdown import MarkItDown
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

md = MarkItDown()

def safe_convert(filepath):
    """帶有錯誤處理的檔案轉換。"""
    try:
        result = md.convert(filepath)
        output = Path(filepath).stem + ".md"

        with open(output, "w") as f:
            f.write(result.text_content)

        logger.info(f"Successfully converted {filepath}")
        return True

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return False

    except ValueError as e:
        logger.error(f"Invalid file format for {filepath}: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error converting {filepath}: {e}")
        return False

# 使用範例
files = ["paper.pdf", "data.xlsx", "slides.pptx"]
results = [safe_convert(f) for f in files]

print(f"Successfully converted {sum(results)}/{len(files)} files")
```

## 進階使用案例

### 自訂元資料擷取

```python
from markitdown import MarkItDown
import re
from datetime import datetime

md = MarkItDown()

def convert_with_metadata(filepath):
    result = md.convert(filepath)

    # 從內容擷取元資料
    metadata = {
        "file": filepath,
        "title": result.title,
        "converted_at": datetime.now().isoformat(),
        "word_count": len(result.text_content.split()),
        "char_count": len(result.text_content)
    }

    # 嘗試尋找作者
    author_match = re.search(r'(?:Author|By):\s*(.+?)(?:\n|$)', result.text_content)
    if author_match:
        metadata["author"] = author_match.group(1).strip()

    # 建立格式化輸出
    output = f"""---
title: {metadata['title']}
author: {metadata.get('author', 'Unknown')}
source: {metadata['file']}
converted: {metadata['converted_at']}
words: {metadata['word_count']}
---

{result.text_content}
"""

    return output, metadata

# 使用範例
content, meta = convert_with_metadata("paper.pdf")
print(meta)
```

### 依格式特定處理

```python
from markitdown import MarkItDown
from pathlib import Path

md = MarkItDown()

def process_by_format(filepath):
    path = Path(filepath)
    result = md.convert(filepath)

    if path.suffix == '.pdf':
        # 添加 PDF 特定元資料
        output = f"# PDF Document: {path.stem}\n\n"
        output += result.text_content

    elif path.suffix == '.xlsx':
        # 添加表格計數
        table_count = result.text_content.count('|---')
        output = f"# Excel Data: {path.stem}\n\n"
        output += f"**Tables**: {table_count}\n\n"
        output += result.text_content

    elif path.suffix == '.pptx':
        # 添加投影片計數
        slide_count = result.text_content.count('## Slide')
        output = f"# Presentation: {path.stem}\n\n"
        output += f"**Slides**: {slide_count}\n\n"
        output += result.text_content

    else:
        output = result.text_content

    return output

# 使用範例
content = process_by_format("presentation.pptx")
print(content)
```

