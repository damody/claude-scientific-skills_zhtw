# 引用文獻品質檢查清單

使用此檢查清單確保您的引用文獻在最終提交前是準確、完整且格式正確的。

## 提交前檢查清單

### ✓ 後設資料準確性

- [ ] 所有作者姓名正確且格式正確
- [ ] 文章標題與實際出版物相符
- [ ] 期刊/會議名稱完整（除非需要否則不縮寫）
- [ ] 出版年份準確
- [ ] 卷號和期號正確
- [ ] 頁碼範圍準確

### ✓ 必填欄位

- [ ] 所有 @article 條目包含：author、title、journal、year
- [ ] 所有 @book 條目包含：author/editor、title、publisher、year
- [ ] 所有 @inproceedings 條目包含：author、title、booktitle、year
- [ ] 現代論文（2000 年以後）可用時包含 DOI
- [ ] 所有條目都有唯一的引用鍵

### ✓ DOI 驗證

- [ ] 所有 DOI 格式正確（10.XXXX/...）
- [ ] DOI 正確解析到文章
- [ ] BibTeX 欄位中無 DOI 前綴（無「doi:」或「https://doi.org/」）
- [ ] CrossRef 的後設資料與您的 BibTeX 條目相符
- [ ] 執行：`python scripts/validate_citations.py references.bib --check-dois`

### ✓ 格式一致性

- [ ] 頁碼範圍使用雙連字符 (--) 而非單連字符 (-)
- [ ] pages 欄位中無「pp.」前綴
- [ ] 作者姓名使用「and」分隔（不是分號或 & 符號）
- [ ] 標題中的大小寫受保護（{AlphaFold}、{CRISPR} 等）
- [ ] 如包含月份，使用標準縮寫
- [ ] 引用鍵遵循一致的格式

### ✓ 重複檢測

- [ ] 參考書目中無重複的 DOI
- [ ] 無重複的引用鍵
- [ ] 無近似重複的標題
- [ ] 可用時將預印本更新為已發表版本
- [ ] 執行：`python scripts/validate_citations.py references.bib`

### ✓ 特殊字元

- [ ] 重音字元格式正確（例如 {\"u} 表示 ü）
- [ ] 數學符號使用 LaTeX 命令
- [ ] 化學公式格式正確
- [ ] 無未轉義的特殊字元（%、&、$、# 等）

### ✓ BibTeX 語法

- [ ] 所有條目的大括號 {} 平衡
- [ ] 欄位之間用逗號分隔
- [ ] 每個條目的最後一個欄位後無逗號
- [ ] 有效的條目類型（@article、@book 等）
- [ ] 執行：`python scripts/validate_citations.py references.bib`

### ✓ 檔案組織

- [ ] 參考書目按邏輯順序排序（按年份、作者或鍵）
- [ ] 全程格式一致
- [ ] 條目之間無格式不一致
- [ ] 執行：`python scripts/format_bibtex.py references.bib --sort year`

## 自動化驗證

### 步驟 1：格式化和清理

```bash
python scripts/format_bibtex.py references.bib \
  --deduplicate \
  --sort year \
  --descending \
  --output clean_references.bib
```

**功能**：
- 移除重複項
- 標準化格式
- 修復常見問題（頁碼範圍、DOI 格式等）
- 按年份排序（最新優先）

### 步驟 2：驗證

```bash
python scripts/validate_citations.py clean_references.bib \
  --check-dois \
  --report validation_report.json \
  --verbose
```

**功能**：
- 檢查必填欄位
- 驗證 DOI 解析
- 檢測重複項
- 驗證語法
- 產生詳細報告

### 步驟 3：檢閱報告

```bash
cat validation_report.json
```

**處理以下問題**：
- **錯誤**：必須修復（遺失的欄位、損壞的 DOI、語法錯誤）
- **警告**：應該修復（遺失的建議欄位、格式問題）
- **重複項**：移除或合併

### 步驟 4：最終檢查

```bash
python scripts/validate_citations.py clean_references.bib --verbose
```

**目標**：零錯誤，最少警告

## 手動檢閱檢查清單

### 關鍵引用文獻（前 10-20 個最重要的）

對於您最重要的引用文獻，手動驗證：

- [ ] 造訪 DOI 連結並確認是正確的文章
- [ ] 對照實際出版物檢查作者姓名
- [ ] 驗證年份與出版日期相符
- [ ] 確認期刊/會議名稱正確
- [ ] 檢查卷號/頁碼是否相符

### 需注意的常見問題

**遺失資訊**：
- [ ] 2000 年後發表的論文無 DOI
- [ ] 期刊文章缺少卷號或頁碼
- [ ] 書籍缺少出版商
- [ ] 會議論文缺少會議地點

**格式錯誤**：
- [ ] 頁碼範圍使用單連字符（123-145 → 123--145）
- [ ] 作者列表中使用 & 符號（Smith & Jones → Smith and Jones）
- [ ] 標題中未保護的縮寫（DNA → {DNA}）
- [ ] DOI 包含 URL 前綴（https://doi.org/10.xxx → 10.xxx）

**後設資料不符**：
- [ ] 作者姓名與出版物不同
- [ ] 年份是線上優先而非紙本出版
- [ ] 期刊名稱縮寫了但應該使用全名
- [ ] 卷號/期號互換

**重複項**：
- [ ] 同一論文使用不同引用鍵引用
- [ ] 預印本和已發表版本都被引用
- [ ] 會議論文和期刊版本都被引用

## 特定領域檢查

### 生物醫學科學

- [ ] 可用時包含 PubMed Central ID (PMCID)
- [ ] MeSH 詞彙適當（如使用）
- [ ] 包含臨床試驗註冊號（如適用）
- [ ] 所有治療/藥物的參考文獻準確引用

### 電腦科學

- [ ] 預印本包含 arXiv ID
- [ ] 會議論文正確引用（不只是「NeurIPS」）
- [ ] 軟體/資料集引用包含版本號
- [ ] GitHub 連結穩定且永久

### 一般科學

- [ ] 資料可用性聲明正確引用
- [ ] 識別並移除撤回的論文
- [ ] 檢查預印本是否有已發表版本
- [ ] 如關鍵，參考補充材料

## 最終提交前步驟

### 提交前 1 週

- [ ] 執行包含 DOI 檢查的完整驗證
- [ ] 修復所有錯誤和關鍵警告
- [ ] 手動驗證前 10-20 個最重要的引用文獻
- [ ] 檢查是否有任何撤回的論文

### 提交前 3 天

- [ ] 任何手動編輯後重新執行驗證
- [ ] 確保所有文內引用都有對應的參考書目條目
- [ ] 確保所有參考書目條目都在文中被引用
- [ ] 檢查引用文獻樣式是否符合期刊要求

### 提交前 1 天

- [ ] 最終驗證檢查
- [ ] LaTeX 編譯成功無警告
- [ ] PDF 正確呈現所有引用文獻
- [ ] 參考書目以正確格式顯示
- [ ] 無占位符引用文獻（Smith et al. XXXX）

### 提交日

- [ ] 最後一次驗證執行
- [ ] 無重新驗證的最後一刻編輯
- [ ] 參考書目檔案包含在提交套件中
- [ ] 文中參考的圖表與參考書目相符

## 品質指標

### 優秀的參考書目

- ✓ 100% 的條目有 DOI（現代論文）
- ✓ 零驗證錯誤
- ✓ 零遺失的必填欄位
- ✓ 零損壞的 DOI
- ✓ 零重複項
- ✓ 全程格式一致
- ✓ 所有引用文獻都經過手動抽查

### 可接受的參考書目

- ✓ 90%+ 的現代條目有 DOI
- ✓ 零高嚴重性錯誤
- ✓ 僅有輕微警告（例如遺失的建議欄位）
- ✓ 關鍵引用文獻已手動驗證
- ✓ 編譯成功無錯誤

### 需要改進

- ✗ 近期論文缺少 DOI
- ✗ 高嚴重性驗證錯誤
- ✗ 損壞或不正確的 DOI
- ✗ 重複條目
- ✗ 格式不一致
- ✗ 編譯警告或錯誤

## 緊急修復

如果您在最後一刻發現問題：

### 損壞的 DOI

```bash
# 尋找正確的 DOI
# 選項 1：搜尋 CrossRef
# https://www.crossref.org/

# 選項 2：在出版商網站搜尋
# 選項 3：Google Scholar

# 重新擷取後設資料
python scripts/extract_metadata.py --doi CORRECT_DOI
```

### 遺失的資訊

```bash
# 從 DOI 擷取
python scripts/extract_metadata.py --doi 10.xxxx/yyyy

# 或從 PMID（生物醫學）
python scripts/extract_metadata.py --pmid 12345678

# 或從 arXiv
python scripts/extract_metadata.py --arxiv 2103.12345
```

### 重複條目

```bash
# 自動移除重複項
python scripts/format_bibtex.py references.bib \
  --deduplicate \
  --output fixed_references.bib
```

### 格式錯誤

```bash
# 自動修復常見問題
python scripts/format_bibtex.py references.bib \
  --output fixed_references.bib

# 然後驗證
python scripts/validate_citations.py fixed_references.bib
```

## 長期最佳實務

### 研究期間

- [ ] 找到引用文獻時立即添加到參考書目檔案
- [ ] 使用 DOI 立即擷取後設資料
- [ ] 每添加 10-20 筆後驗證
- [ ] 將參考書目檔案納入版本控制

### 寫作期間

- [ ] 邊寫邊引用
- [ ] 使用一致的引用鍵
- [ ] 不要延遲添加參考文獻
- [ ] 每週驗證

### 提交前

- [ ] 預留 2-3 天進行引用文獻清理
- [ ] 不要等到最後一天
- [ ] 盡可能自動化
- [ ] 手動驗證關鍵引用文獻

## 工具快速參考

### 擷取後設資料

```bash
# 從 DOI
python scripts/doi_to_bibtex.py 10.1038/nature12345

# 從多個來源
python scripts/extract_metadata.py \
  --doi 10.1038/nature12345 \
  --pmid 12345678 \
  --arxiv 2103.12345 \
  --output references.bib
```

### 驗證

```bash
# 基本驗證
python scripts/validate_citations.py references.bib

# 包含 DOI 檢查（慢但徹底）
python scripts/validate_citations.py references.bib --check-dois

# 產生報告
python scripts/validate_citations.py references.bib \
  --report validation.json \
  --verbose
```

### 格式化和清理

```bash
# 格式化並修復問題
python scripts/format_bibtex.py references.bib

# 移除重複項並排序
python scripts/format_bibtex.py references.bib \
  --deduplicate \
  --sort year \
  --descending \
  --output clean_refs.bib
```

## 總結

**最低要求**：
1. 執行 `format_bibtex.py --deduplicate`
2. 執行 `validate_citations.py`
3. 修復所有錯誤
4. 成功編譯

**建議**：
1. 格式化、去重和排序
2. 使用 `--check-dois` 驗證
3. 修復所有錯誤和警告
4. 手動驗證頂級引用文獻
5. 修復後重新驗證

**最佳實務**：
1. 在整個研究過程中驗證
2. 一致地使用自動化工具
3. 保持參考書目乾淨有序
4. 記錄任何特殊情況
5. 提交前 1-3 天進行最終驗證

**記住**：引用文獻錯誤會對您的學術形象產生負面影響。花時間確保準確性是值得的！
