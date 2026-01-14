# 使用範例與工作流程

## 完整工作流程範例

### 範例 1：會議演示套件

**場景**：為重要的會議演示準備網站、海報和影片。

**用戶請求**："我需要為我的 NeurIPS 論文提交創建完整的演示套件。生成網站、海報和影片演示。"

**工作流程**：

```bash
# 步驟 1：組織論文檔案
mkdir -p input/neurips2025_paper
cp main.tex input/neurips2025_paper/
cp -r figures/ input/neurips2025_paper/
cp -r tables/ input/neurips2025_paper/
cp bibliography.bib input/neurips2025_paper/

# 步驟 2：生成所有組件
python pipeline_all.py \
  --input-dir input/neurips2025_paper \
  --output-dir output/ \
  --model-choice 1 \
  --generate-website \
  --generate-poster \
  --generate-video \
  --poster-width-inches 48 \
  --poster-height-inches 36 \
  --enable-logo-search

# 步驟 3：審核輸出
ls -R output/neurips2025_paper/
# - website/index.html
# - poster/poster_final.pdf
# - video/final_video.mp4
```

**輸出**：
- 展示研究的互動式網站
- 4'×3' 會議海報（可印刷）
- 12 分鐘演示影片
- 處理時間：約 45 分鐘（無虛擬主持人）

---

### 範例 2：預印本快速網站

**場景**：為 bioRxiv 預印本創建可探索的首頁。

**用戶請求**："將我的基因組學預印本轉換為互動式網站，作為 bioRxiv 提交的配套。"

**工作流程**：

```bash
# 使用 PDF 輸入（無 LaTeX 可用）
python pipeline_all.py \
  --input-dir papers/genomics_preprint/ \
  --output-dir output/genomics_web/ \
  --model-choice 1 \
  --generate-website

# 部署到 GitHub Pages 或個人伺服器
cd output/genomics_web/website/
# 添加 bioRxiv 論文連結、資料儲存庫、程式碼
# 上傳到託管服務
```

**提示**：
- 包含 bioRxiv DOI 連結
- 添加 GitHub 儲存庫連結
- 包含資料可用性區塊
- 如可能，嵌入互動式視覺化

---

### 範例 3：期刊投稿影片摘要

**場景**：為鼓勵多媒體提交的期刊創建影片摘要。

**用戶請求**："為我的 Nature Communications 投稿生成 5 分鐘的影片摘要。"

**工作流程**：

```bash
# 生成聚焦關鍵發現的簡潔影片
python pipeline_light.py \
  --model_name_t gpt-4.1 \
  --model_name_v gpt-4.1 \
  --result_dir output/video_abstract/ \
  --paper_latex_root papers/nature_comms/ \
  --video-duration 300 \
  --slides-per-minute 3

# 可選：添加自定義片頭/片尾投影片
# 可選：在引言中加入虛擬主持人
```

**輸出**：
- 5 分鐘影片摘要
- 聚焦視覺結果
- 清晰易懂的旁白
- 期刊規格格式

---

### 範例 4：多論文網站生成

**場景**：為研究團隊的多篇論文創建網站。

**用戶請求**："為我們實驗室今年發表的所有 5 篇論文生成網站。"

**工作流程**：

```bash
# 組織論文
mkdir -p batch_input/
# 創建子目錄：paper1/、paper2/、paper3/、paper4/、paper5/
# 每個目錄包含其 LaTeX 源檔案

# 批次處理
python pipeline_all.py \
  --input-dir batch_input/ \
  --output-dir batch_output/ \
  --model-choice 1 \
  --generate-website \
  --enable-logo-search

# 創建：
# batch_output/paper1/website/
# batch_output/paper2/website/
# batch_output/paper3/website/
# batch_output/paper4/website/
# batch_output/paper5/website/
```

**最佳實踐**：
- 使用一致的命名慣例
- 大批量處理可過夜執行
- 審核每個網站的準確性
- 部署到統一的實驗室網站

---

### 範例 5：虛擬會議海報

**場景**：為虛擬會議創建具有互動元素的數位海報。

**用戶請求**："為虛擬 ISMB 會議創建海報，包含可點擊的程式碼和資料連結。"

**工作流程**：

```bash
# 生成帶有 QR 碼和連結的海報
python pipeline_all.py \
  --input-dir papers/ismb_submission/ \
  --output-dir output/ismb_poster/ \
  --model-choice 1 \
  --generate-poster \
  --poster-width-inches 48 \
  --poster-height-inches 36 \
  --enable-qr-codes

# 手動添加 QR 碼到：
# - GitHub 儲存庫
# - 互動式結果儀表板
# - 補充資料
# - 影片演示
```

**數位增強**：
- 具有嵌入超連結的 PDF
- 用於虛擬平台的高解析度 PNG
- 帶有影片連結的獨立 PDF 供下載

---

### 範例 6：推廣影片片段

**場景**：為社交媒體創建簡短的推廣影片。

**用戶請求**："為我們的 Cell 論文生成 2 分鐘的精華影片用於 Twitter。"

**工作流程**：

```bash
# 生成簡短、引人入勝的影片
python pipeline_light.py \
  --model_name_t gpt-4.1 \
  --model_name_v gpt-4.1 \
  --result_dir output/promo_video/ \
  --paper_latex_root papers/cell_paper/ \
  --video-duration 120 \
  --presentation-style public

# 後期處理：
# - 提取 30 秒的 Twitter 精華片段
# - 為靜音觀看添加字幕
# - 針對社交媒體最佳化檔案大小
```

**社交媒體最佳化**：
- 正方形格式（1:1）用於 Instagram
- 橫式格式（16:9）用於 Twitter/LinkedIn
- 直式格式（9:16）用於 TikTok/Stories
- 添加關鍵發現的文字覆蓋

---

## 常見使用案例模式

### 模式 1：LaTeX 論文 → 完整套件

**輸入**：LaTeX 源檔案及所有資源
**輸出**：網站 + 海報 + 影片
**時間**：45-90 分鐘
**最適合**：重要出版物、會議演示

```bash
python pipeline_all.py \
  --input-dir [latex_dir] \
  --output-dir [output_dir] \
  --model-choice 1 \
  --generate-website \
  --generate-poster \
  --generate-video
```

---

### 模式 2：PDF → 互動式網站

**輸入**：已發表的 PDF 論文
**輸出**：可探索的網站
**時間**：15-30 分鐘
**最適合**：發表後推廣、預印本增強

```bash
python pipeline_all.py \
  --input-dir [pdf_dir] \
  --output-dir [output_dir] \
  --model-choice 1 \
  --generate-website
```

---

### 模式 3：LaTeX → 會議海報

**輸入**：LaTeX 論文
**輸出**：可印刷海報（自定義尺寸）
**時間**：10-20 分鐘
**最適合**：會議海報展示

```bash
python pipeline_all.py \
  --input-dir [latex_dir] \
  --output-dir [output_dir] \
  --model-choice 1 \
  --generate-poster \
  --poster-width-inches [width] \
  --poster-height-inches [height]
```

---

### 模式 4：LaTeX → 演示影片

**輸入**：LaTeX 論文
**輸出**：旁白演示影片
**時間**：20-60 分鐘（無虛擬主持人）
**最適合**：影片摘要、線上演示、課程材料

```bash
python pipeline_light.py \
  --model_name_t gpt-4.1 \
  --model_name_v gpt-4.1 \
  --result_dir [output_dir] \
  --paper_latex_root [latex_dir]
```

---

## 平台特定輸出

### Twitter/X 推廣內容

系統針對數字資料夾名稱自動檢測 Twitter 目標：

```bash
# 創建 Twitter 最佳化內容
mkdir -p input/001_twitter_post/
# 系統生成英文推廣內容
```

**生成輸出**：
- 簡短、引人入勝的摘要
- 關鍵圖像精華
- 標籤建議
- 推文串格式

---

### 小紅書內容

對於中文社交媒體，使用字母數字資料夾名稱：

```bash
# 創建小紅書最佳化內容
mkdir -p input/xhs_genomics/
# 系統生成中文推廣內容
```

**生成輸出**：
- 中文內容
- 平台適當的格式
- 視覺優先的呈現
- 參與度最佳化

---

## 常見場景故障排除

### 場景：大型論文（>50 頁）

**挑戰**：處理時間和內容選擇
**解決方案**：
```bash
# 選項 1：聚焦關鍵章節
# 編輯 LaTeX 註解掉較不重要的章節

# 選項 2：分部處理
# 生成概述網站
# 為方法/結果生成獨立的詳細影片

# 選項 3：使用較快模型進行初次處理
# 審核並使用更好的模型重新生成關鍵組件
```

---

### 場景：複雜數學內容

**挑戰**：方程式可能無法完美渲染
**解決方案**：
- 使用 LaTeX 輸入（非 PDF）以獲得最佳方程式處理
- 審核生成內容的方程式準確性
- 如需要手動調整複雜方程式
- 考慮對關鍵方程式使用圖像截圖

---

### 場景：非標準論文結構

**挑戰**：論文不遵循標準 IMRAD 格式
**解決方案**：
- 在論文元資料中提供自定義章節指導
- 審核生成的結構並調整
- 使用更強大的模型（GPT-4.1）以獲得更好的適應性
- 考慮在 LaTeX 註解中手動標注章節

---

### 場景：有限的 API 預算

**挑戰**：在維持品質的同時降低成本
**解決方案**：
```bash
# 對簡單論文使用 GPT-3.5-turbo
python pipeline_all.py \
  --input-dir [paper_dir] \
  --output-dir [output_dir] \
  --model-choice 3

# 僅生成所需組件
# 僅網站（最便宜）
# 僅海報（中等）
# 無虛擬主持人的影片（中等）
```

---

### 場景：緊迫的截止日期

**挑戰**：需要快速輸出
**解決方案**：
```bash
# 如有多篇論文可並行處理
# 使用更快的模型（GPT-3.5-turbo）
# 先生成最重要的組件
# 跳過可選功能（標誌搜索、虛擬主持人）

python pipeline_light.py \
  --model_name_t gpt-3.5-turbo \
  --model_name_v gpt-3.5-turbo \
  --result_dir [output_dir] \
  --paper_latex_root [latex_dir]
```

**優先順序**：
1. 網站（最快、最通用）
2. 海報（中等速度、印刷截止日期）
3. 影片（最慢、可稍後生成）

---

## 品質最佳化提示

### 獲得最佳網站結果
1. 使用包含所有資源的 LaTeX 輸入
2. 包含高解析度圖像
3. 確保論文具有清晰的章節結構
4. 啟用標誌搜索以獲得專業外觀
5. 審核並測試所有互動元素

### 獲得最佳海報結果
1. 提供高解析度圖像（300+ DPI）
2. 指定所需的確切海報尺寸
3. 包含機構品牌資訊
4. 使用專業配色方案
5. 在完整海報前測試列印小預覽

### 獲得最佳影片結果
1. 使用 LaTeX 以獲得最清晰的內容提取
2. 適當指定目標時長
3. 在生成影片前審核腳本
4. 選擇適當的演示風格
5. 測試音訊品質和節奏

### 獲得最佳整體結果
1. 從乾淨、組織良好的 LaTeX 源檔案開始
2. 使用 GPT-4 或 GPT-4.1 獲得最高品質
3. 在最終確定前審核所有輸出
4. 對需要調整的組件進行迭代
5. 結合組件以獲得連貫的演示套件
