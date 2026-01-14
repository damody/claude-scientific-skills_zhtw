# 引用文獻驗證指南

驗證 BibTeX 檔案中引用文獻準確性、完整性和格式的完整指南。

## 概述

引用文獻驗證確保：
- 所有引用文獻準確且完整
- DOI 正確解析
- 必填欄位存在
- 無重複條目
- 格式和語法正確
- 連結可存取

驗證應在以下時機執行：
- 擷取後設資料後
- 手稿提交前
- 手動編輯 BibTeX 檔案後
- 定期維護參考書目時

## 驗證類別

### 1. DOI 驗證

**目的**：確保 DOI 有效且正確解析。

#### 檢查項目

**DOI 格式**：
```
有效：   10.1038/s41586-021-03819-2
有效：   10.1126/science.aam9317
無效：   10.1038/invalid
無效：   doi:10.1038/...（BibTeX 中應省略「doi:」前綴）
```

**DOI 解析**：
- DOI 應透過 https://doi.org/ 解析
- 應重新導向到實際文章
- 不應返回 404 或錯誤

**後設資料一致性**：
- CrossRef 後設資料應與 BibTeX 相符
- 作者姓名應一致
- 標題應相符
- 年份應相符

#### 如何驗證

**手動檢查**：
1. 從 BibTeX 複製 DOI
2. 造訪 https://doi.org/10.1038/nature12345
3. 驗證重新導向到正確文章
4. 檢查後設資料相符

**自動檢查**（建議）：
```bash
python scripts/validate_citations.py references.bib --check-dois
```

**處理流程**：
1. 從 BibTeX 檔案擷取所有 DOI
2. 對每個 DOI 查詢 doi.org 解析器
3. 查詢 CrossRef API 取得後設資料
4. 比較後設資料與 BibTeX 條目
5. 報告差異

#### 常見問題

**損壞的 DOI**：
- DOI 中有打字錯誤
- 出版商更改了 DOI（罕見）
- 文章被撤回
- 解決方案：從出版商網站尋找正確的 DOI

**後設資料不符**：
- BibTeX 有舊的/不正確的資訊
- 解決方案：從 CrossRef 重新擷取後設資料

**遺失的 DOI**：
- 較舊的文章可能沒有 DOI
- 2000 年前的出版物可接受
- 改為添加 URL 或 PMID

### 2. 必填欄位

**目的**：確保所有必要資訊都存在。

#### 各條目類型的必填欄位

**@article**：
```bibtex
author   % 必填
title    % 必填
journal  % 必填
year     % 必填
volume   % 強烈建議
pages    % 強烈建議
doi      % 現代論文強烈建議
```

**@book**：
```bibtex
author 或 editor  % 必填（至少一個）
title            % 必填
publisher        % 必填
year             % 必填
isbn             % 建議
```

**@inproceedings**：
```bibtex
author     % 必填
title      % 必填
booktitle  % 必填（會議/論文集名稱）
year       % 必填
pages      % 建議
```

**@incollection**（書籍章節）：
```bibtex
author     % 必填
title      % 必填（章節標題）
booktitle  % 必填（書名）
publisher  % 必填
year       % 必填
editor     % 建議
pages      % 建議
```

**@phdthesis**：
```bibtex
author  % 必填
title   % 必填
school  % 必填
year    % 必填
```

**@misc**（預印本、資料集等）：
```bibtex
author  % 必填
title   % 必填
year    % 必填
howpublished  % 建議（bioRxiv、Zenodo 等）
doi 或 url    % 至少需要一個
```

#### 驗證腳本

```bash
python scripts/validate_citations.py references.bib --check-required-fields
```

**輸出**：
```
Error: Entry 'Smith2024' missing required field 'journal'
Error: Entry 'Doe2023' missing required field 'year'
Warning: Entry 'Jones2022' missing recommended field 'volume'
```

### 3. 作者姓名格式

**目的**：確保作者姓名格式一致、正確。

#### 正確格式

**建議的 BibTeX 格式**：
```bibtex
author = {Last1, First1 and Last2, First2 and Last3, First3}
```

**範例**：
```bibtex
% 正確
author = {Smith, John}
author = {Smith, John A.}
author = {Smith, John Andrew}
author = {Smith, John and Doe, Jane}
author = {Smith, John and Doe, Jane and Johnson, Mary}

% 多位作者
author = {Smith, John and Doe, Jane and others}

% 不正確
author = {John Smith}  % First Last 格式（不建議）
author = {Smith, J.; Doe, J.}  % 分號分隔（錯誤）
author = {Smith J, Doe J}  % 遺失逗號
```

#### 特殊情況

**後綴（Jr.、III 等）**：
```bibtex
author = {King, Jr., Martin Luther}
```

**多姓氏（連字符）**：
```bibtex
author = {Smith-Jones, Mary}
```

**Van、von、de 等**：
```bibtex
author = {van der Waals, Johannes}
author = {de Broglie, Louis}
```

**組織作為作者**：
```bibtex
author = {{World Health Organization}}
% 雙大括號視為單一作者
```

#### 驗證檢查

**自動化驗證**：
```bash
python scripts/validate_citations.py references.bib --check-authors
```

**檢查項目**：
- 正確的分隔符（and，而非 &、; 等）
- 逗號位置
- 空的作者欄位
- 格式錯誤的姓名

### 4. 資料一致性

**目的**：確保所有欄位包含有效、合理的值。

#### 年份驗證

**有效年份**：
```bibtex
year = {2024}    % 當前/近期
year = {1953}    % Watson & Crick DNA 結構（歷史性）
year = {1665}    % Hooke 的 Micrographia（非常舊）
```

**無效年份**：
```bibtex
year = {24}      % 兩位數（模糊）
year = {202}     % 打字錯誤
year = {2025}    % 未來（除非已接受/付印中）
year = {0}       % 明顯錯誤
```

**檢查**：
- 四位數
- 合理範圍（1600-當前+1）
- 不全為零

#### 卷號/期號驗證

```bibtex
volume = {123}      % 數值
volume = {12}       % 有效
number = {3}        % 有效
number = {S1}       % 增刊期（有效）
```

**無效**：
```bibtex
volume = {Vol. 123}  % 應只是數字
number = {Issue 3}   % 應只是數字
```

#### 頁碼範圍驗證

**正確格式**：
```bibtex
pages = {123--145}    % 長破折號（雙連字符）
pages = {e0123456}    % PLOS 風格文章 ID
pages = {123}         % 單頁
```

**不正確格式**：
```bibtex
pages = {123-145}     % 單連字符（使用 --）
pages = {pp. 123-145} % 移除「pp.」
pages = {123–145}     % Unicode 長破折號（可能導致問題）
```

#### URL 驗證

**檢查**：
- URL 可存取（返回 200 狀態）
- 可用時使用 HTTPS
- 無明顯打字錯誤
- 永久連結（非臨時）

**有效**：
```bibtex
url = {https://www.nature.com/articles/nature12345}
url = {https://arxiv.org/abs/2103.14030}
```

**可疑**：
```bibtex
url = {http://...}  % HTTP 而非 HTTPS
url = {file:///...} % 本地檔案路徑
url = {bit.ly/...}  % URL 縮短服務（非永久）
```

### 5. 重複檢測

**目的**：尋找並移除重複條目。

#### 重複類型

**完全重複**（相同 DOI）：
```bibtex
@article{Smith2024a,
  doi = {10.1038/nature12345},
  ...
}

@article{Smith2024b,
  doi = {10.1038/nature12345},  % 相同 DOI！
  ...
}
```

**近似重複**（相似標題/作者）：
```bibtex
@article{Smith2024,
  title = {Machine Learning for Drug Discovery},
  ...
}

@article{Smith2024method,
  title = {Machine learning for drug discovery},  % 相同，不同大小寫
  ...
}
```

**預印本 + 已發表**：
```bibtex
@misc{Smith2023arxiv,
  title = {AlphaFold Results},
  howpublished = {arXiv},
  ...
}

@article{Smith2024,
  title = {AlphaFold Results},  % 同一論文，現已發表
  journal = {Nature},
  ...
}
% 僅保留已發表版本
```

#### 檢測方法

**按 DOI**（最可靠）：
- 相同 DOI = 完全重複
- 保留一個，移除另一個

**按標題相似度**：
- 標準化：小寫，移除標點符號
- 計算相似度（例如 Levenshtein 距離）
- 如相似度 >90% 則標記

**按作者-年份-標題**：
- 相同第一作者 + 年份 + 相似標題
- 可能重複

**自動檢測**：
```bash
python scripts/validate_citations.py references.bib --check-duplicates
```

**輸出**：
```
Warning: Possible duplicate entries:
  - Smith2024a (DOI: 10.1038/nature12345)
  - Smith2024b (DOI: 10.1038/nature12345)
  Recommendation: Keep one entry, remove the other.
```

### 6. 格式和語法

**目的**：確保有效的 BibTeX 語法。

#### 常見語法錯誤

**遺失逗號**：
```bibtex
@article{Smith2024,
  author = {Smith, John}   % 遺失逗號！
  title = {Title}
}
% 應為：
  author = {Smith, John},  % 每個欄位後有逗號
```

**大括號不平衡**：
```bibtex
title = {Title with {Protected} Text  % 遺失右大括號
% 應為：
title = {Title with {Protected} Text}
```

**條目遺失右大括號**：
```bibtex
@article{Smith2024,
  author = {Smith, John},
  title = {Title}
  % 遺失右大括號！
% 應以此結尾：
}
```

**引用鍵中的無效字元**：
```bibtex
@article{Smith&Doe2024,  % 引用鍵中不允許 &
  ...
}
% 使用：
@article{SmithDoe2024,
  ...
}
```

#### BibTeX 語法規則

**條目結構**：
```bibtex
@TYPE{citationkey,
  field1 = {value1},
  field2 = {value2},
  ...
  fieldN = {valueN}
}
```

**引用鍵**：
- 英數字和一些標點符號（-、_、.、:）
- 無空格
- 區分大小寫
- 檔案內唯一

**欄位值**：
- 用 {大括號} 或 "引號" 括起
- 複雜文字建議用大括號
- 數字可不加引號：`year = 2024`

**特殊字元**：
- `{` 和 `}` 用於分組
- `\` 用於 LaTeX 命令
- 保護大小寫：`{AlphaFold}`
- 重音：`{\"u}`、`{\'e}`、`{\aa}`

#### 驗證

```bash
python scripts/validate_citations.py references.bib --check-syntax
```

**檢查**：
- 有效的 BibTeX 結構
- 大括號平衡
- 正確的逗號
- 有效的條目類型
- 唯一的引用鍵

## 驗證工作流程

### 步驟 1：基本驗證

執行全面驗證：

```bash
python scripts/validate_citations.py references.bib
```

**檢查所有項目**：
- DOI 解析
- 必填欄位
- 作者格式
- 資料一致性
- 重複項
- 語法

### 步驟 2：檢閱報告

檢查驗證報告：

```json
{
  "total_entries": 150,
  "valid_entries": 140,
  "errors": [
    {
      "entry": "Smith2024",
      "error": "missing_required_field",
      "field": "journal",
      "severity": "high"
    },
    {
      "entry": "Doe2023",
      "error": "invalid_doi",
      "doi": "10.1038/broken",
      "severity": "high"
    }
  ],
  "warnings": [
    {
      "entry": "Jones2022",
      "warning": "missing_recommended_field",
      "field": "volume",
      "severity": "medium"
    }
  ],
  "duplicates": [
    {
      "entries": ["Smith2024a", "Smith2024b"],
      "reason": "same_doi",
      "doi": "10.1038/nature12345"
    }
  ]
}
```

### 步驟 3：修復問題

**高優先順序**（錯誤）：
1. 添加遺失的必填欄位
2. 修復損壞的 DOI
3. 移除重複項
4. 更正語法錯誤

**中優先順序**（警告）：
1. 添加建議欄位
2. 改進作者格式
3. 修復頁碼範圍

**低優先順序**：
1. 標準化格式
2. 添加 URL 以提高可存取性

### 步驟 4：自動修復

對安全的更正使用自動修復：

```bash
python scripts/validate_citations.py references.bib \
  --auto-fix \
  --output fixed_references.bib
```

**自動修復可以**：
- 修復頁碼範圍格式（- 改為 --）
- 從 pages 移除「pp.」
- 標準化作者分隔符
- 修復常見語法錯誤
- 標準化欄位順序

**自動修復無法**：
- 添加遺失的資訊
- 尋找正確的 DOI
- 決定保留哪個重複項
- 修復語義錯誤

### 步驟 5：手動檢閱

檢閱自動修復的檔案：
```bash
# 檢查變更內容
diff references.bib fixed_references.bib

# 檢閱有錯誤的特定條目
grep -A 10 "Smith2024" fixed_references.bib
```

### 步驟 6：重新驗證

修復後驗證：

```bash
python scripts/validate_citations.py fixed_references.bib --verbose
```

應顯示：
```
✓ All DOIs valid
✓ All required fields present
✓ No duplicates found
✓ Syntax valid
✓ 150/150 entries valid
```

## 驗證檢查清單

最終提交前使用此檢查清單：

### DOI 驗證
- [ ] 所有 DOI 正確解析
- [ ] BibTeX 和 CrossRef 之間的後設資料相符
- [ ] 無損壞或無效的 DOI

### 完整性
- [ ] 所有條目都有必填欄位
- [ ] 現代論文（2000 年以後）有 DOI
- [ ] 作者格式正確
- [ ] 期刊/會議名稱正確

### 一致性
- [ ] 年份是 4 位數數字
- [ ] 頁碼範圍使用 -- 而非 -
- [ ] 卷號/期號是數值
- [ ] URL 可存取

### 重複項
- [ ] 無相同 DOI 的條目
- [ ] 無近似重複的標題
- [ ] 預印本已更新為已發表版本

### 格式
- [ ] 有效的 BibTeX 語法
- [ ] 大括號平衡
- [ ] 正確的逗號
- [ ] 唯一的引用鍵

### 最終檢查
- [ ] 參考書目編譯無錯誤
- [ ] 文中所有引用都在參考書目中
- [ ] 參考書目中所有條目都在文中被引用
- [ ] 引用文獻樣式符合期刊要求

## 最佳實務

### 1. 儘早並經常驗證

```bash
# 擷取後
python scripts/extract_metadata.py --doi ... --output refs.bib
python scripts/validate_citations.py refs.bib

# 手動編輯後
python scripts/validate_citations.py refs.bib

# 提交前
python scripts/validate_citations.py refs.bib --strict
```

### 2. 使用自動化工具

不要手動驗證 - 使用腳本：
- 更快
- 更全面
- 捕捉人工遺漏的錯誤
- 產生報告

### 3. 保留備份

```bash
# 自動修復前
cp references.bib references_backup.bib

# 執行自動修復
python scripts/validate_citations.py references.bib \
  --auto-fix \
  --output references_fixed.bib

# 檢閱變更
diff references.bib references_fixed.bib

# 如滿意，替換
mv references_fixed.bib references.bib
```

### 4. 先修復高優先順序

**優先順序**：
1. 語法錯誤（阻止編譯）
2. 遺失的必填欄位（不完整的引用文獻）
3. 損壞的 DOI（損壞的連結）
4. 重複項（混淆、浪費空間）
5. 遺失的建議欄位
6. 格式不一致

### 5. 記錄例外

對於無法修復的條目：

```bibtex
@article{Old1950,
  author = {Smith, John},
  title = {Title},
  journal = {Obscure Journal},
  year = {1950},
  volume = {12},
  pages = {34--56},
  note = {DOI not available for publications before 2000}
}
```

### 6. 根據期刊要求驗證

不同期刊有不同要求：
- 引用文獻樣式（編號、作者-年份）
- 縮寫（期刊名稱）
- 最大參考文獻數量
- 格式（BibTeX、EndNote、手動）

查看期刊作者指南！

## 常見驗證問題

### 問題 1：後設資料不符

**問題**：BibTeX 顯示 2023，CrossRef 顯示 2024。

**原因**：
- 線上優先與紙本出版
- 更正/更新
- 擷取錯誤

**解決方案**：
1. 檢查實際文章
2. 使用更新/準確的日期
3. 更新 BibTeX 條目
4. 重新驗證

### 問題 2：特殊字元

**問題**：LaTeX 編譯因特殊字元失敗。

**原因**：
- 重音字元（é、ü、ñ）
- 化學公式（H₂O）
- 數學符號（α、β、±）

**解決方案**：
```bibtex
% 使用 LaTeX 命令
author = {M{\"u}ller, Hans}  % Müller
title = {Study of H\textsubscript{2}O}  % H₂O
% 或使用 UTF-8 配合適當的 LaTeX 套件
```

### 問題 3：不完整的擷取

**問題**：擷取的後設資料遺失欄位。

**原因**：
- 來源未提供所有後設資料
- 擷取錯誤
- 不完整的記錄

**解決方案**：
1. 檢查原始文章
2. 手動添加遺失的欄位
3. 使用替代來源（PubMed vs CrossRef）

### 問題 4：無法找到重複項

**問題**：同一論文出現兩次，未被檢測到。

**原因**：
- 不同的 DOI（應該很少見）
- 不同的標題（縮寫、打字錯誤）
- 不同的引用鍵

**解決方案**：
- 手動搜尋作者 + 年份
- 檢查相似標題
- 手動移除

## 總結

驗證確保引用文獻品質：

✓ **準確性**：DOI 解析，後設資料正確
✓ **完整性**：所有必填欄位存在
✓ **一致性**：全程格式正確
✓ **無重複**：每篇論文只引用一次
✓ **有效語法**：BibTeX 編譯無錯誤

**最終提交前務必驗證**！

使用自動化工具：
```bash
python scripts/validate_citations.py references.bib
```

遵循工作流程：
1. 擷取後設資料
2. 驗證
3. 修復錯誤
4. 重新驗證
5. 提交
