# 安裝與配置

## 系統需求

### 硬體需求
- **GPU**：NVIDIA A6000（最低 48GB）用於具有虛擬主持人功能的影片生成
- **CPU**：建議使用多核心處理器進行 PDF 處理和文件轉換
- **RAM**：最低 16GB，大型論文建議 32GB

### 軟體需求
- **Python**：3.11 或更高版本
- **Conda**：用於依賴項隔離的環境管理器
- **LibreOffice**：用於文件格式轉換（PDF 到 PPTX 等）
- **Poppler 工具**：用於 PDF 處理和操作

## 安裝步驟

### 1. 克隆儲存庫
```bash
git clone https://github.com/YuhangChen1/Paper2All.git
cd Paper2All
```

### 2. 創建 Conda 環境
```bash
conda create -n paper2all python=3.11
conda activate paper2all
```

### 3. 安裝依賴項
```bash
pip install -r requirements.txt
```

### 4. 安裝系統依賴項

**Ubuntu/Debian：**
```bash
sudo apt-get install libreoffice poppler-utils
```

**macOS：**
```bash
brew install libreoffice poppler
```

**Windows：**
- 從 https://www.libreoffice.org/ 下載並安裝 LibreOffice
- 從 https://github.com/oschwartz10612/poppler-windows 下載並安裝 Poppler

## API 配置

在專案根目錄中創建 `.env` 檔案，包含以下憑證：

### 必要的 API 金鑰

**選項 1：OpenAI API**
```
OPENAI_API_KEY=your_openai_api_key_here
```

**選項 2：OpenRouter API**（OpenAI 的替代方案）
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 可選的 API 金鑰

**Google Search API**（用於自動標誌發現）
```
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

## 模型配置

系統支援多個 LLM 後端：

### 支援的模型
- GPT-4（建議用於最佳品質）
- GPT-4.1（最新版本）
- GPT-3.5-turbo（更快速，成本較低）
- 透過 OpenRouter 使用 Claude 模型
- 其他 OpenRouter 支援的模型

### 模型選擇

使用 `--model-choice` 參數或 `--model_name_t` 和 `--model_name_v` 參數指定模型：
- 模型選擇 1：所有組件使用 GPT-4
- 模型選擇 2：所有組件使用 GPT-4.1
- 自定義：為文字和視覺處理指定不同的模型

## 驗證

測試安裝：

```bash
python pipeline_all.py --help
```

如果成功，您應該看到包含所有可用選項的說明選單。

## 故障排除

### 常見問題

**1. 找不到 LibreOffice**
- 確保 LibreOffice 已安裝並在系統 PATH 中
- 嘗試運行 `libreoffice --version` 進行驗證

**2. 找不到 Poppler 工具**
- 使用 `pdftoppm -v` 驗證安裝
- 如需要，將 Poppler bin 目錄添加到 PATH

**3. 影片生成的 GPU/CUDA 錯誤**
- 確保 NVIDIA 驅動程式是最新的
- 驗證 CUDA 工具包已安裝
- 使用 `nvidia-smi` 檢查 GPU 記憶體

**4. API 金鑰錯誤**
- 驗證 `.env` 檔案位於專案根目錄
- 檢查 API 金鑰是否有效且有足夠的額度
- 確保 `.env` 中的金鑰周圍沒有額外的空格或引號

## 目錄結構

安裝後，組織您的工作區：

```
Paper2All/
├── .env                  # API 憑證
├── input/               # 在此放置您的論文檔案
│   └── paper_name/      # 每篇論文放在自己的目錄中
│       └── main.tex     # LaTeX 源檔案或 PDF
├── output/              # 生成的輸出
│   └── paper_name/
│       ├── website/     # 生成的網站檔案
│       ├── video/       # 生成的影片檔案
│       └── poster/      # 生成的海報檔案
└── ...
```
