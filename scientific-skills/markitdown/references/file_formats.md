# 檔案格式支援

本文件提供 MarkItDown 支援的每種檔案格式的詳細資訊。

## 文件格式

### PDF (.pdf)

**功能**：
- 文字擷取
- 表格檢測
- 元資料擷取
- 掃描文件的 OCR（需要依賴項）

**依賴項**：
```bash
pip install 'markitdown[pdf]'
```

**最適用於**：
- 科學論文
- 報告
- 書籍
- 表單

**限制**：
- 複雜佈局可能無法保持完美格式
- 掃描的 PDF 需要 OCR 設定
- 某些 PDF 功能（註解、表單）可能無法轉換

**範例**：
```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("research_paper.pdf")
print(result.text_content)
```

**使用 Azure 文件智慧增強**：
```python
md = MarkItDown(docintel_endpoint="https://YOUR-ENDPOINT.cognitiveservices.azure.com/")
result = md.convert("complex_layout.pdf")
```

---

### Microsoft Word (.docx)

**功能**：
- 文字擷取
- 表格轉換
- 標題層級
- 列表格式
- 基本文字格式（粗體、斜體）

**依賴項**：
```bash
pip install 'markitdown[docx]'
```

**最適用於**：
- 研究論文
- 報告
- 文件
- 手稿

**保留的元素**：
- 標題（轉換為 Markdown 標題）
- 表格（轉換為 Markdown 表格）
- 列表（項目符號和編號）
- 基本格式（粗體、斜體）
- 段落

**範例**：
```python
result = md.convert("manuscript.docx")
```

---

### PowerPoint (.pptx)

**功能**：
- 投影片內容擷取
- 演講者備註
- 表格擷取
- 圖片描述（使用 AI）

**依賴項**：
```bash
pip install 'markitdown[pptx]'
```

**最適用於**：
- 簡報
- 講座投影片
- 研討會演講

**輸出格式**：
```markdown
# Slide 1: Title

Content from slide 1...

**Notes**: Speaker notes appear here

---

# Slide 2: Next Topic

...
```

**使用 AI 圖片描述**：
```python
from openai import OpenAI

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4o")
result = md.convert("presentation.pptx")
```

---

### Excel (.xlsx, .xls)

**功能**：
- 工作表擷取
- 表格格式化
- 資料保留
- 公式值（計算後）

**依賴項**：
```bash
pip install 'markitdown[xlsx]'  # 現代 Excel
pip install 'markitdown[xls]'   # 舊版 Excel
```

**最適用於**：
- 資料表
- 研究資料
- 統計結果
- 實驗資料

**輸出格式**：
```markdown
# Sheet: Results

| Sample | Control | Treatment | P-value |
|--------|---------|-----------|---------|
| 1      | 10.2    | 12.5      | 0.023   |
| 2      | 9.8     | 11.9      | 0.031   |
```

**範例**：
```python
result = md.convert("experimental_data.xlsx")
```

---

## 圖片格式

### 圖片 (.jpg, .jpeg, .png, .gif, .webp)

**功能**：
- EXIF 元資料擷取
- OCR 文字擷取
- AI 驅動的圖片描述

**依賴項**：
```bash
pip install 'markitdown[all]'  # 包含圖片支援
```

**最適用於**：
- 掃描文件
- 圖表和圖形
- 科學圖表
- 含文字的照片

**無 AI 的輸出**：
```markdown
![Image](image.jpg)

**EXIF Data**:
- Camera: Canon EOS 5D
- Date: 2024-01-15
- Resolution: 4000x3000
```

**使用 AI 的輸出**：
```python
from openai import OpenAI

client = OpenAI()
md = MarkItDown(
    llm_client=client,
    llm_model="gpt-4o",
    llm_prompt="Describe this scientific diagram in detail"
)
result = md.convert("graph.png")
```

**用於文字擷取的 OCR**：
需要 Tesseract OCR：
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

---

## 音訊格式

### 音訊 (.wav, .mp3)

**功能**：
- 元資料擷取
- 語音轉文字轉錄
- 時長和技術資訊

**依賴項**：
```bash
pip install 'markitdown[audio-transcription]'
```

**最適用於**：
- 講座錄音
- 訪談
- 播客
- 會議錄音

**輸出格式**：
```markdown
# Audio: interview.mp3

**Metadata**:
- Duration: 45:32
- Bitrate: 320kbps
- Sample Rate: 44100Hz

**Transcription**:
[Transcribed text appears here...]
```

**範例**：
```python
result = md.convert("lecture.mp3")
```

---

## 網頁格式

### HTML (.html, .htm)

**功能**：
- 乾淨的 HTML 到 Markdown 轉換
- 連結保留
- 表格轉換
- 列表格式化

**最適用於**：
- 網頁
- 文件
- 部落格文章
- 線上文章

**輸出格式**：保留連結和結構的乾淨 Markdown

**範例**：
```python
result = md.convert("webpage.html")
```

---

### YouTube 網址

**功能**：
- 擷取影片轉錄
- 擷取影片元資料
- 字幕下載

**依賴項**：
```bash
pip install 'markitdown[youtube-transcription]'
```

**最適用於**：
- 教育影片
- 講座
- 演講
- 教學

**範例**：
```python
result = md.convert("https://www.youtube.com/watch?v=VIDEO_ID")
```

---

## 資料格式

### CSV (.csv)

**功能**：
- 自動表格轉換
- 分隔符號檢測
- 標題保留

**輸出格式**：Markdown 表格

**範例**：
```python
result = md.convert("data.csv")
```

**輸出**：
```markdown
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Value1  | Value2  | Value3  |
```

---

### JSON (.json)

**功能**：
- 結構化表示
- 美觀格式化
- 巢狀資料視覺化

**最適用於**：
- API 回應
- 設定檔
- 資料匯出

**範例**：
```python
result = md.convert("data.json")
```

---

### XML (.xml)

**功能**：
- 結構保留
- 屬性擷取
- 格式化輸出

**最適用於**：
- 設定檔
- 資料交換
- 結構化文件

**範例**：
```python
result = md.convert("config.xml")
```

---

## 壓縮格式

### ZIP (.zip)

**功能**：
- 迭代處理壓縮檔內容
- 個別轉換每個檔案
- 在輸出中維護目錄結構

**最適用於**：
- 文件集合
- 專案壓縮檔
- 批次轉換

**輸出格式**：
```markdown
# Archive: documents.zip

## File: document1.pdf
[Content from document1.pdf...]

---

## File: document2.docx
[Content from document2.docx...]
```

**範例**：
```python
result = md.convert("archive.zip")
```

---

## 電子書格式

### EPUB (.epub)

**功能**：
- 完整文字擷取
- 章節結構
- 元資料擷取

**最適用於**：
- 電子書
- 數位出版物
- 長篇內容

**輸出格式**：保留章節結構的 Markdown

**範例**：
```python
result = md.convert("book.epub")
```

---

## 其他格式

### Outlook 郵件 (.msg)

**功能**：
- 電子郵件內容擷取
- 附件列表
- 元資料（寄件者、收件者、主旨、日期）

**依賴項**：
```bash
pip install 'markitdown[outlook]'
```

**最適用於**：
- 電子郵件存檔
- 通訊記錄

**範例**：
```python
result = md.convert("message.msg")
```

---

## 格式特定技巧

### PDF 最佳實踐

1. **對複雜佈局使用 Azure 文件智慧**：
   ```python
   md = MarkItDown(docintel_endpoint="endpoint_url")
   ```

2. **對於掃描的 PDF，確保已設定 OCR**：
   ```bash
   brew install tesseract  # macOS
   ```

3. **在轉換前分割非常大的 PDF** 以獲得更好的效能

### PowerPoint 最佳實踐

1. **對視覺內容使用 AI**：
   ```python
   md = MarkItDown(llm_client=client, llm_model="gpt-4o")
   ```

2. **檢查演講者備註** - 它們會包含在輸出中

3. **複雜動畫不會被擷取** - 僅限靜態內容

### Excel 最佳實踐

1. **大型試算表** 可能需要較長的轉換時間

2. **公式會轉換為其計算值**

3. **多個工作表** 都會包含在輸出中

4. **圖表會變成文字描述**（使用 AI 可獲得更好的描述）

### 圖片最佳實踐

1. **使用 AI 獲得有意義的描述**：
   ```python
   md = MarkItDown(
       llm_client=client,
       llm_model="gpt-4o",
       llm_prompt="Describe this scientific figure in detail"
   )
   ```

2. **對於含大量文字的圖片，確保** 已安裝 OCR 依賴項

3. **高解析度圖片** 可能需要較長的處理時間

### 音訊最佳實踐

1. **清晰的音訊** 會產生更好的轉錄

2. **長錄音** 可能需要大量時間

3. **考慮分割長音訊檔案** 以加快處理速度

---

## 不支援的格式

如果您需要轉換不支援的格式：

1. **建立自訂轉換器**（參見 `api_reference.md`）
2. **在 GitHub 上尋找外掛**（#markitdown-plugin）
3. **預先轉換為支援的格式**（例如將 .rtf 轉換為 .docx）

---

## 格式檢測

MarkItDown 自動從以下來源檢測格式：

1. **檔案副檔名**（主要方法）
2. **MIME 類型**（後備）
3. **檔案簽名**（magic bytes，後備）

**覆蓋檢測**：
```python
# 強制使用特定格式
result = md.convert("file_without_extension", file_extension=".pdf")

# 使用串流時
with open("file", "rb") as f:
    result = md.convert_stream(f, file_extension=".pdf")
```

