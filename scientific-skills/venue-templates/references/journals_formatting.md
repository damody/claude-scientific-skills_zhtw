# 期刊格式要求

各學科主要科學期刊的完整格式要求與投稿指南。

**最後更新**：2024

---

## Nature Portfolio

### Nature

**期刊類型**：頂級多學科科學期刊
**出版商**：Nature Publishing Group
**影響因子（Impact Factor）**：約 64（每年略有不同）

**格式要求**：
- **長度**：文章約 3,000 字（不含 Methods、References、Figure Legends）
- **結構**：Title、Authors、Affiliations、Abstract（≤200 字）、Main text、Methods、References、Acknowledgements、Author Contributions、Competing Interests、Figure Legends
- **格式**：投稿為單欄（最終發表版為雙欄）
- **字型**：任何標準字型（Times、Arial、Helvetica），12pt
- **行距**：雙倍行距
- **邊距**：2.5 cm（1 英寸）四邊
- **頁碼**：所有頁面需標註
- **引用**：以上標數字順序標註¹'²'³
- **參考文獻**：Nature 格式（期刊名稱縮寫）
  - 格式：Author, A. A., Author, B. B. & Author, C. C. Article title. *Journal Abbrev.* **vol**, pages (year).
  - 範例：Watson, J. D. & Crick, F. H. C. Molecular structure of nucleic acids. *Nature* **171**, 737–738 (1953).
- **圖片**：
  - 格式：TIFF、EPS、PDF（向量圖優先）
  - 解析度：照片 300-600 dpi，線條圖 1000 dpi
  - 色彩：RGB 或 CMYK
  - 尺寸：適合單欄（89 mm）或雙欄（183 mm）寬度
  - 圖說：另外提供，不嵌入圖片中
- **表格**：可編輯格式（Word、Excel），非圖片
- **補充資料（Supplementary Info）**：無限制，PDF 格式優先

**LaTeX 模板**：`assets/journals/nature_article.tex`

**作者指南**：https://www.nature.com/nature/for-authors

---

### Nature Communications

**期刊類型**：開放取用（Open Access）多學科期刊
**出版商**：Nature Publishing Group

**格式要求**：
- **長度**：無嚴格限制（通常 5,000-8,000 字）
- **結構**：與 Nature 相同（Title、Abstract、Main text、Methods、References 等）
- **格式**：單欄
- **字型**：Times New Roman、Arial 或類似字型，12pt
- **行距**：雙倍行距
- **邊距**：2.5 cm 四邊
- **引用**：以上標數字順序標註
- **參考文獻**：Nature 格式（與 Nature 相同）
- **圖片**：與 Nature 要求相同
- **表格**：與 Nature 要求相同
- **開放取用**：所有文章均為開放取用（需支付 APC 文章處理費）

**LaTeX 模板**：`assets/journals/nature_communications.tex`

---

### Nature Methods、Nature Biotechnology、Nature Machine Intelligence

**格式**：與 Nature Communications 相同（Nature 系列期刊共用類似格式）

**學科特定說明**：
- **Nature Methods**：強調方法創新與驗證
- **Nature Biotechnology**：著重生物技術應用與轉化
- **Nature Machine Intelligence**：跨學科的 AI/ML 應用

---

## Science 系列

### Science

**期刊類型**：頂級多學科科學期刊
**出版商**：美國科學促進會（AAAS）

**格式要求**：
- **長度**：
  - Research Articles：2,500 字（僅正文，不含參考文獻/圖片）
  - Reports：最多 2,500 字
- **結構**：Title、Authors、Affiliations、Abstract（≤125 字）、Main text、Materials and Methods、References、Acknowledgments、Supplementary Materials
- **格式**：投稿為單欄
- **字型**：Times New Roman，12pt
- **行距**：雙倍行距
- **邊距**：1 英寸四邊
- **引用**：以括號數字順序標註 (1, 2, 3)
- **參考文獻**：Science 格式（主要參考文獻不含文章標題，移至補充資料）
  - 格式：A. Author, B. Author, *Journal Abbrev.* **vol**, pages (year).
  - 範例：J. D. Watson, F. H. C. Crick, *Nature* **171**, 737 (1953).
- **圖片**：
  - 格式：PDF、EPS、TIFF
  - 解析度：最低 300 dpi
  - 色彩：RGB
  - 尺寸：最大寬度 9 cm（單欄）或 18.3 cm（雙欄）
  - 圖片計入頁數限制
- **表格**：包含在正文中或作為獨立檔案
- **補充資料**：允許大量補充材料

**LaTeX 模板**：`assets/journals/science_article.tex`

**作者指南**：https://www.science.org/content/page/instructions-authors

---

### Science Advances

**期刊類型**：開放取用多學科期刊
**出版商**：AAAS

**格式要求**：
- **長度**：無嚴格字數限制（但鼓勵精簡寫作）
- **結構**：與 Science 類似（較有彈性）
- **格式**：單欄
- **字型**：Times New Roman，12pt
- **引用**：括號數字標註
- **參考文獻**：Science 格式
- **圖片**：與 Science 相同
- **開放取用**：所有文章為開放取用

**LaTeX 模板**：`assets/journals/science_advances.tex`

---

## PLOS（公共科學圖書館）

### PLOS ONE

**期刊類型**：開放取用多學科期刊
**出版商**：Public Library of Science

**格式要求**：
- **長度**：無最大長度限制
- **結構**：Title、Authors、Affiliations、Abstract、Introduction、Materials and Methods、Results、Discussion、Conclusions（可選）、References、Supporting Information
- **格式**：可編輯檔案（LaTeX、Word、RTF）
- **字型**：Times、Arial 或 Helvetica，10-12pt
- **行距**：雙倍行距
- **邊距**：1 英寸（2.54 cm）四邊
- **頁碼**：必須標註
- **引用**：Vancouver 格式，以方括號數字標註 [1]、[2]、[3]
- **參考文獻**：Vancouver/NLM 格式
  - 格式：Author AA, Author BB, Author CC. Article title. Journal Abbrev. Year;vol(issue):pages. doi:xx.xxxx
  - 範例：Watson JD, Crick FHC. Molecular structure of nucleic acids. Nature. 1953;171(4356):737-738.
- **圖片**：
  - 格式：TIFF、EPS、PDF、PNG
  - 解析度：300-600 dpi
  - 色彩：RGB
  - 圖說：在參考文獻後的正文中提供
- **表格**：可編輯格式，每頁一個
- **資料可用性（Data Availability）**：需提供聲明
- **開放取用**：所有文章為開放取用（需支付 APC）

**LaTeX 模板**：`assets/journals/plos_one.tex`

**作者指南**：https://journals.plos.org/plosone/s/submission-guidelines

---

### PLOS Biology、PLOS Computational Biology 等

**格式**：與 PLOS ONE 類似，有學科特定變化

**主要差異**：
- PLOS Biology：更具選擇性，強調廣泛重要性
- PLOS Comp Bio：著重計算方法與模型

---

## Cell Press

### Cell

**期刊類型**：頂級生物學期刊
**出版商**：Cell Press（Elsevier）

**格式要求**：
- **長度**：
  - Articles：約 5,000 字（不含 Methods、References）
  - Short Articles：約 2,500 字
- **結構**：Summary（≤150 字）、Keywords、Introduction、Results、Discussion、Experimental Procedures、Acknowledgments、Author Contributions、Declaration of Interests、References
- **格式**：雙倍行距
- **字型**：12pt
- **邊距**：1 英寸四邊
- **引用**：作者-年份格式（Smith et al., 2023）
- **參考文獻**：Cell 格式
  - 格式：Author, A.A., and Author, B.B. (Year). Title. *Journal* vol, pages.
  - 範例：Watson, J.D., and Crick, F.H. (1953). Molecular structure of nucleic acids. *Nature* 171, 737-738.
- **圖片**：
  - 格式：TIFF、EPS 優先
  - 解析度：照片 300 dpi，線條圖 1000 dpi
  - 色彩：RGB 或 CMYK
  - 常見多面板圖片
- **表格**：可編輯格式
- **eTOC Blurb**：需提供 30-50 字摘要
- **圖形摘要（Graphical Abstract）**：必須提供

**LaTeX 模板**：`assets/journals/cell_article.tex`

**作者指南**：https://www.cell.com/cell/authors

---

### Neuron、Immunity、Molecular Cell、Developmental Cell

**格式**：與 Cell 類似，有學科特定期望

---

## IEEE Transactions

### IEEE Transactions on [各種主題]

**期刊類型**：工程與電腦科學期刊
**出版商**：電機電子工程師學會（IEEE）

**格式要求**：
- **長度**：依不同 transactions 而異（通常最終格式 8-12 頁）
- **結構**：Abstract、Index Terms、Introduction、[正文章節]、Conclusion、Acknowledgment、References、Biographies
- **格式**：雙欄
- **字型**：Times New Roman，10pt
- **欄間距**：0.17 英寸（4.23 mm）
- **邊距**：
  - 上：19 mm（0.75 英寸）
  - 下：25 mm（1 英寸）
  - 側邊：17 mm（0.67 英寸）
- **引用**：以方括號數字標註 [1]、[2]、[3]
- **參考文獻**：IEEE 格式
  - 格式：[1] A. A. Author, "Title of paper," *Journal Abbrev.*, vol. x, no. x, pp. xxx-xxx, Mon. Year.
  - 範例：[1] J. D. Watson and F. H. C. Crick, "Molecular structure of nucleic acids," *Nature*, vol. 171, pp. 737-738, Apr. 1953.
- **圖片**：
  - 格式：EPS、PDF（向量圖）、TIFF（點陣圖）
  - 解析度：線條圖 600-1200 dpi，灰階/彩色 300 dpi
  - 色彩：線上版 RGB，印刷版如需則用 CMYK
  - 位置：欄的頂部或底部
- **表格**：LaTeX table 環境，置於頂部/底部
- **方程式**：連續編號

**LaTeX 模板**：`assets/journals/ieee_trans.tex`

**作者指南**：https://journals.ieeeauthorcenter.ieee.org/

---

### IEEE Access

**期刊類型**：開放取用多學科工程期刊
**出版商**：IEEE

**格式**：與 IEEE Transactions 類似
- **長度**：無頁數限制
- **開放取用**：所有文章為開放取用
- **快速發表**：審查速度比 Transactions 更快

**LaTeX 模板**：`assets/journals/ieee_access.tex`

---

## ACM Publications

### ACM Transactions

**期刊類型**：電腦科學 transactions
**出版商**：計算機協會（ACM）

**格式要求**：
- **長度**：無嚴格限制
- **結構**：Abstract、CCS Concepts、Keywords、ACM Reference Format、Introduction、[正文]、Conclusion、Acknowledgments、References
- **格式**：雙欄（最終版），投稿可用單欄
- **字型**：依模板而定（通常 9-10pt）
- **文件類別**：使用 `acmart` LaTeX 文件類別
- **引用**：數字標註 [1] 或作者-年份，依場所而定
- **參考文獻**：ACM 格式
  - 格式：Author. Year. Title. Journal vol, issue (Year), pages. DOI
  - 範例：James D. Watson and Francis H. C. Crick. 1953. Molecular structure of nucleic acids. Nature 171, 4356 (1953), 737-738. https://doi.org/10.1038/171737a0
- **圖片**：EPS、PDF（向量圖優先）、高解析度點陣圖
- **CCS Concepts**：必須提供（ACM 計算分類系統）
- **關鍵字**：必須提供

**LaTeX 模板**：`assets/journals/acm_article.tex`

**作者指南**：https://www.acm.org/publications/authors

---

## Springer 期刊

### 一般 Springer 期刊

**出版商**：Springer Nature

**格式要求**：
- **長度**：依期刊而異（請查閱特定期刊）
- **格式**：投稿為單欄（LaTeX 或 Word）
- **字型**：10-12pt
- **行距**：雙倍或 1.5
- **引用**：數字標註或作者-年份（依期刊而異）
- **參考文獻**：Springer 格式（類似 Vancouver 或作者-年份）
  - 數字：Author AA, Author BB (Year) Title. Journal vol:pages
  - 作者-年份：Author AA, Author BB (Year) Title. Journal vol:pages
- **圖片**：TIFF、EPS、PDF；300+ dpi
- **表格**：可編輯格式
- **文件類別**：許多 Springer 期刊使用 `svjour3`

**LaTeX 模板**：`assets/journals/springer_article.tex`

**作者指南**：依特定期刊而異

---

## Elsevier 期刊

### 一般 Elsevier 期刊

**出版商**：Elsevier

**格式要求**：
- **長度**：依期刊差異很大
- **格式**：單欄（LaTeX 或 Word）
- **字型**：12pt
- **行距**：雙倍行距
- **引用**：數字標註或作者-年份（請查閱期刊指南）
- **參考文獻**：格式依期刊而異（Harvard、Vancouver、數字標註）
  - 請查閱特定期刊的「Guide for Authors」
- **圖片**：TIFF、EPS；300+ dpi
- **表格**：可編輯格式
- **文件類別**：`elsarticle` LaTeX 類別

**LaTeX 模板**：`assets/journals/elsevier_article.tex`

**作者指南**：https://www.elsevier.com/authors（選擇特定期刊）

---

## BMC 期刊

### BMC Biology、BMC Bioinformatics 等

**出版商**：BioMed Central（Springer Nature）

**格式要求**：
- **長度**：無最大長度限制
- **結構**：Abstract（結構化）、Keywords、Background、[Methods/Results/Discussion]、Conclusions、Abbreviations、Declarations（Ethics、Consent、Availability、Competing interests、Funding、Authors' contributions、Acknowledgements）、References
- **格式**：單欄
- **字型**：Arial 或 Times，12pt
- **行距**：雙倍
- **引用**：Vancouver 格式，以方括號數字標註 [1]
- **參考文獻**：Vancouver/NLM 格式
- **圖片**：TIFF、EPS、PNG；300+ dpi
- **表格**：可編輯
- **開放取用**：所有 BMC 期刊為開放取用
- **資料可用性**：需提供聲明

**LaTeX 模板**：`assets/journals/bmc_article.tex`

**作者指南**：https://www.biomedcentral.com/getpublished

---

## Frontiers 期刊

### Frontiers in [各種主題]

**出版商**：Frontiers Media

**格式要求**：
- **長度**：依文章類型而異（Research Article 約 12 頁，Brief Research Report 約 4 頁）
- **結構**：Abstract、Keywords、Introduction、Materials and Methods、Results、Discussion、Conclusion、Data Availability Statement、Ethics Statement、Author Contributions、Funding、Acknowledgments、Conflict of Interest、References
- **格式**：單欄
- **字型**：Times New Roman，12pt
- **行距**：雙倍
- **引用**：數字標註（Frontiers 格式）
- **參考文獻**：Frontiers 格式
  - 格式：Author A., Author B., Author C. (Year). Title. *Journal Abbrev.* vol:pages. doi
  - 範例：Watson J. D., Crick F. H. C. (1953). Molecular structure of nucleic acids. *Nature* 171:737-738. doi:10.1038/171737a0
- **圖片**：TIFF、EPS；最低 300 dpi
- **表格**：可編輯
- **開放取用**：所有 Frontiers 期刊為開放取用
- **圖說**：詳細說明，每張圖最多 350 字

**LaTeX 模板**：`assets/journals/frontiers_article.tex`

**作者指南**：https://www.frontiersin.org/guidelines/author-guidelines

---

## 專業期刊

### PNAS（美國國家科學院院刊）

**格式要求**：
- **長度**：6 頁（正文、圖片、表格合計）
- **摘要**：最多 250 字
- **重要性聲明（Significance Statement）**：最多 120 字（必須提供）
- **結構**：Abstract、Significance、Main text、Materials and Methods、Acknowledgments、References
- **格式**：單欄
- **引用**：數字標註
- **參考文獻**：PNAS 格式
- **LaTeX 類別**：`pnas-new`

**LaTeX 模板**：`assets/journals/pnas_article.tex`

---

### Physical Review Letters (PRL)

**出版商**：美國物理學會（APS）

**格式要求**：
- **長度**：4 頁（包含圖片和參考文獻）
- **格式**：雙欄（REVTeX 4.2）
- **摘要**：不超過 600 字元
- **引用**：數字標註
- **參考文獻**：APS 格式
- **文件類別**：`revtex4-2`

**LaTeX 模板**：`assets/journals/prl_article.tex`

---

### New England Journal of Medicine (NEJM)

**格式要求**：
- **長度**：Original Articles 約 3,000 字
- **結構**：Abstract（結構化，250 字）、Introduction、Methods、Results、Discussion、References
- **格式**：雙倍行距
- **引用**：數字標註
- **參考文獻**：NEJM 格式（修改版 Vancouver）
- **圖片**：高解析度，專業品質
- **Word 投稿優先**（LaTeX 較不常見）

---

### The Lancet

**格式要求**：
- **長度**：Articles 約 3,000 字
- **摘要**：結構化，300 字
- **結構**：Panel（摘要框）、Introduction、Methods、Results、Discussion、References
- **引用**：數字標註
- **參考文獻**：Lancet 格式（修改版 Vancouver）
- **Word 優先**投稿

---

## 快速參考表

| 期刊 | 最大長度 | 格式 | 引用 | 模板 |
|---------|-----------|--------|-----------|----------|
| **Nature** | 約 3,000 字 | 單欄 | 上標¹ | `nature_article.tex` |
| **Science** | 2,500 字 | 單欄 | (1) 括號 | `science_article.tex` |
| **PLOS ONE** | 無限制 | 單欄 | [1] Vancouver | `plos_one.tex` |
| **Cell** | 約 5,000 字 | 雙倍行距 | (Author, year) | `cell_article.tex` |
| **IEEE Trans** | 8-12 頁 | 雙欄 | [1] IEEE | `ieee_trans.tex` |
| **ACM Trans** | 可變 | 雙欄 | [1] 或作者-年份 | `acm_article.tex` |
| **Springer** | 可變 | 單欄 | 數字/作者-年份 | `springer_article.tex` |
| **BMC** | 無限制 | 單欄 | [1] Vancouver | `bmc_article.tex` |
| **Frontiers** | 約 12 頁 | 單欄 | 數字標註 | `frontiers_article.tex` |

---

## 注意事項

1. **務必查閱官方指南**：期刊要求會變更；投稿前請確認
2. **模板時效性**：這些模板定期更新，但可能落後於官方變更
3. **補充材料**：大多數期刊允許大量補充材料
4. **預印本政策**：查閱期刊的預印本政策（大多數允許 arXiv、bioRxiv）
5. **開放取用選項**：許多訂閱期刊提供付費開放取用選項
6. **LaTeX vs. Word**：大多數期刊接受兩者；數學內容較多時優先使用 LaTeX

## 獲取官方模板

許多期刊提供官方 LaTeX 模板：
- **Nature**：從期刊網站下載
- **IEEE**：IEEEtran 類別（廣泛可用）
- **ACM**：acmart 類別（CTAN）
- **Elsevier**：elsarticle 類別（CTAN）
- **Springer**：svjour3 類別（期刊網站）

請查閱期刊的「For Authors」或「Submit」頁面以獲取最新模板。
