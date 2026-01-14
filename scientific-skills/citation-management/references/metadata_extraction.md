# 後設資料擷取指南

使用各種 API 和服務從 DOI、PMID、arXiv ID 和 URL 擷取準確引用後設資料的完整指南。

## 概述

準確的後設資料對正確引用至關重要。本指南涵蓋：
- 識別論文識別碼（DOI、PMID、arXiv ID）
- 查詢後設資料 API（CrossRef、PubMed、arXiv、DataCite）
- 各條目類型的必填 BibTeX 欄位
- 處理邊緣案例和特殊情況
- 驗證擷取的後設資料

## 論文識別碼

### DOI（數位物件識別碼）

**格式**：`10.XXXX/suffix`

**範例**：
```
10.1038/s41586-021-03819-2    # Nature 文章
10.1126/science.aam9317       # Science 文章
10.1016/j.cell.2023.01.001    # Cell 文章
10.1371/journal.pone.0123456  # PLOS ONE 文章
```

**特性**：
- 永久識別碼
- 最可靠的後設資料來源
- 解析到當前位置
- 由出版商指派

**哪裡可以找到**：
- 文章首頁
- 文章網頁
- CrossRef、Google Scholar、PubMed
- 通常在出版商網站上顯眼位置

### PMID（PubMed ID）

**格式**：8 位數數字（通常）

**範例**：
```
34265844
28445112
35476778
```

**特性**：
- 專屬於 PubMed 資料庫
- 僅限生物醫學文獻
- 由 NCBI 指派
- 永久識別碼

**哪裡可以找到**：
- PubMed 搜尋結果
- PubMed 文章頁面
- 通常在文章 PDF 頁尾
- PMC（PubMed Central）頁面

### PMCID（PubMed Central ID）

**格式**：PMC 後接數字

**範例**：
```
PMC8287551
PMC7456789
```

**特性**：
- PMC 中的免費全文文章
- PubMed 文章的子集
- 開放存取或作者手稿

### arXiv ID

**格式**：YYMM.NNNNN 或 archive/YYMMNNN

**範例**：
```
2103.14030        # 新格式（2007 年以後）
2401.12345        # 2024 年提交
arXiv:hep-th/9901001  # 舊格式
```

**特性**：
- 預印本（未經同行評審）
- 物理、數學、電腦科學、定量生物學等
- 版本追蹤（v1、v2 等）
- 免費、開放存取

**哪裡可以找到**：
- arXiv.org
- 通常在發表前被引用
- 論文 PDF 頁首

### 其他識別碼

**ISBN**（書籍）：
```
978-0-12-345678-9
0-123-45678-9
```

**arXiv 類別**：
```
cs.LG    # 電腦科學 - 機器學習
q-bio.QM # 定量生物學 - 定量方法
math.ST  # 數學 - 統計學
```

## 後設資料 API

### CrossRef API

**DOI 的主要來源** - 期刊文章最全面的後設資料。

**基礎 URL**：`https://api.crossref.org/works/`

**無需 API 金鑰**，但建議使用禮貌池：
- 在 User-Agent 中添加電子郵件
- 獲得更好的服務
- 無速率限制

#### 基本 DOI 查詢

**請求**：
```
GET https://api.crossref.org/works/10.1038/s41586-021-03819-2
```

**回應**（簡化）：
```json
{
  "message": {
    "DOI": "10.1038/s41586-021-03819-2",
    "title": ["Article title here"],
    "author": [
      {"given": "John", "family": "Smith"},
      {"given": "Jane", "family": "Doe"}
    ],
    "container-title": ["Nature"],
    "volume": "595",
    "issue": "7865",
    "page": "123-128",
    "published-print": {"date-parts": [[2021, 7, 1]]},
    "publisher": "Springer Nature",
    "type": "journal-article",
    "ISSN": ["0028-0836"]
  }
}
```

#### 可用欄位

**始終存在**：
- `DOI`：數位物件識別碼
- `title`：文章標題（陣列）
- `type`：內容類型（journal-article、book-chapter 等）

**通常存在**：
- `author`：作者物件陣列
- `container-title`：期刊/書名
- `published-print` 或 `published-online`：出版日期
- `volume`、`issue`、`page`：出版詳細資訊
- `publisher`：出版商名稱

**有時存在**：
- `abstract`：文章摘要
- `subject`：主題類別
- `ISSN`：期刊 ISSN
- `ISBN`：書籍 ISBN
- `reference`：參考文獻列表
- `is-referenced-by-count`：引用次數

#### 內容類型

CrossRef `type` 欄位值：
- `journal-article`：期刊文章
- `book-chapter`：書籍章節
- `book`：書籍
- `proceedings-article`：會議論文
- `posted-content`：預印本
- `dataset`：研究資料集
- `report`：技術報告
- `dissertation`：論文/學位論文

### PubMed E-utilities API

**專為生物醫學文獻** - 含 MeSH 詞彙的策展後設資料。

**基礎 URL**：`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

**建議使用 API 金鑰**（免費）：
- 更高的速率限制
- 更好的效能

#### PMID 轉後設資料

**步驟 1：使用 EFetch 取得完整記錄**

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?
  db=pubmed&
  id=34265844&
  retmode=xml&
  api_key=YOUR_KEY
```

**回應**：包含完整後設資料的 XML

**步驟 2：解析 XML**

關鍵欄位：
```xml
<PubmedArticle>
  <MedlineCitation>
    <PMID>34265844</PMID>
    <Article>
      <ArticleTitle>Title here</ArticleTitle>
      <AuthorList>
        <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
      </AuthorList>
      <Journal>
        <Title>Nature</Title>
        <JournalIssue>
          <Volume>595</Volume>
          <Issue>7865</Issue>
          <PubDate><Year>2021</Year></PubDate>
        </JournalIssue>
      </Journal>
      <Pagination><MedlinePgn>123-128</MedlinePgn></Pagination>
      <Abstract><AbstractText>Abstract text here</AbstractText></Abstract>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="doi">10.1038/s41586-021-03819-2</ArticleId>
      <ArticleId IdType="pmc">PMC8287551</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
```

#### PubMed 獨有欄位

**MeSH 詞彙**：控制詞彙
```xml
<MeshHeadingList>
  <MeshHeading>
    <DescriptorName UI="D003920">Diabetes Mellitus</DescriptorName>
  </MeshHeading>
</MeshHeadingList>
```

**出版類型**：
```xml
<PublicationTypeList>
  <PublicationType UI="D016428">Journal Article</PublicationType>
  <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
</PublicationTypeList>
```

**經費資訊**：
```xml
<GrantList>
  <Grant>
    <GrantID>R01-123456</GrantID>
    <Agency>NIAID NIH HHS</Agency>
    <Country>United States</Country>
  </Grant>
</GrantList>
```

### arXiv API

**物理、數學、電腦科學、定量生物學的預印本** - 免費、開放存取。

**基礎 URL**：`http://export.arxiv.org/api/query`

**無需 API 金鑰**

#### arXiv ID 轉後設資料

**請求**：
```
GET http://export.arxiv.org/api/query?id_list=2103.14030
```

**回應**：Atom XML

```xml
<entry>
  <id>http://arxiv.org/abs/2103.14030v2</id>
  <title>Highly accurate protein structure prediction with AlphaFold</title>
  <author><name>John Jumper</name></author>
  <author><name>Richard Evans</name></author>
  <published>2021-03-26T17:47:17Z</published>
  <updated>2021-07-01T16:51:46Z</updated>
  <summary>Abstract text here...</summary>
  <arxiv:doi>10.1038/s41586-021-03819-2</arxiv:doi>
  <category term="q-bio.BM" scheme="http://arxiv.org/schemas/atom"/>
  <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
</entry>
```

#### 關鍵欄位

- `id`：arXiv URL
- `title`：預印本標題
- `author`：作者列表
- `published`：首版日期
- `updated`：最新版日期
- `summary`：摘要
- `arxiv:doi`：如已發表則有 DOI
- `arxiv:journal_ref`：如已發表則有期刊參考
- `category`：arXiv 類別

#### 版本追蹤

arXiv 追蹤版本：
- `v1`：初始提交
- `v2`、`v3` 等：修訂版

**始終檢查**預印本是否已在期刊發表（如可用則使用 DOI）。

### DataCite API

**研究資料集、軟體、其他產出** - 為非傳統學術產出指派 DOI。

**基礎 URL**：`https://api.datacite.org/dois/`

**類似 CrossRef** 但用於資料集、軟體、程式碼等。

**請求**：
```
GET https://api.datacite.org/dois/10.5281/zenodo.1234567
```

**回應**：包含資料集/軟體後設資料的 JSON

## 必填 BibTeX 欄位

### @article（期刊文章）

**必填**：
- `author`：作者姓名
- `title`：文章標題
- `journal`：期刊名稱
- `year`：出版年份

**選用但建議**：
- `volume`：卷號
- `number`：期號
- `pages`：頁碼範圍（例如 123--145）
- `doi`：數位物件識別碼
- `url`：如無 DOI 則用 URL
- `month`：出版月份

**範例**：
```bibtex
@article{Smith2024,
  author  = {Smith, John and Doe, Jane},
  title   = {Novel Approach to Protein Folding},
  journal = {Nature},
  year    = {2024},
  volume  = {625},
  number  = {8001},
  pages   = {123--145},
  doi     = {10.1038/nature12345}
}
```

### @book（書籍）

**必填**：
- `author` 或 `editor`：作者或編輯
- `title`：書名
- `publisher`：出版商名稱
- `year`：出版年份

**選用但建議**：
- `edition`：版次（如非第一版）
- `address`：出版商地點
- `isbn`：ISBN
- `url`：URL
- `series`：叢書名稱

**範例**：
```bibtex
@book{Kumar2021,
  author    = {Kumar, Vinay and Abbas, Abul K. and Aster, Jon C.},
  title     = {Robbins and Cotran Pathologic Basis of Disease},
  publisher = {Elsevier},
  year      = {2021},
  edition   = {10},
  isbn      = {978-0-323-53113-9}
}
```

### @inproceedings（會議論文）

**必填**：
- `author`：作者姓名
- `title`：論文標題
- `booktitle`：會議/論文集名稱
- `year`：年份

**選用但建議**：
- `pages`：頁碼範圍
- `organization`：主辦單位
- `publisher`：出版商
- `address`：會議地點
- `month`：會議月份
- `doi`：如可用則有 DOI

**範例**：
```bibtex
@inproceedings{Vaswani2017,
  author    = {Vaswani, Ashish and Shazeer, Noam and others},
  title     = {Attention is All You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  pages     = {5998--6008},
  volume    = {30}
}
```

### @incollection（書籍章節）

**必填**：
- `author`：章節作者
- `title`：章節標題
- `booktitle`：書名
- `publisher`：出版商名稱
- `year`：出版年份

**選用但建議**：
- `editor`：書籍編輯
- `pages`：章節頁碼範圍
- `chapter`：章節號
- `edition`：版次
- `address`：出版商地點

**範例**：
```bibtex
@incollection{Brown2020,
  author    = {Brown, Peter O. and Botstein, David},
  title     = {Exploring the New World of the Genome with {DNA} Microarrays},
  booktitle = {DNA Microarrays: A Molecular Cloning Manual},
  editor    = {Eisen, Michael B. and Brown, Patrick O.},
  publisher = {Cold Spring Harbor Laboratory Press},
  year      = {2020},
  pages     = {1--45}
}
```

### @phdthesis（博士論文）

**必填**：
- `author`：作者姓名
- `title`：論文標題
- `school`：機構
- `year`：年份

**選用**：
- `type`：類型（例如「PhD dissertation」）
- `address`：機構地點
- `month`：月份
- `url`：URL

**範例**：
```bibtex
@phdthesis{Johnson2023,
  author = {Johnson, Mary L.},
  title  = {Novel Approaches to Cancer Immunotherapy},
  school = {Stanford University},
  year   = {2023},
  type   = {{PhD} dissertation}
}
```

### @misc（預印本、軟體、資料集）

**必填**：
- `author`：作者
- `title`：標題
- `year`：年份

**對於預印本，添加**：
- `howpublished`：儲存庫（例如「bioRxiv」）
- `doi`：預印本 DOI
- `note`：預印本 ID

**範例（預印本）**：
```bibtex
@misc{Zhang2024,
  author       = {Zhang, Yi and Chen, Li and Wang, Hui},
  title        = {Novel Therapeutic Targets in Alzheimer's Disease},
  year         = {2024},
  howpublished = {bioRxiv},
  doi          = {10.1101/2024.01.001},
  note         = {Preprint}
}
```

**範例（軟體）**：
```bibtex
@misc{AlphaFold2021,
  author       = {DeepMind},
  title        = {{AlphaFold} Protein Structure Database},
  year         = {2021},
  howpublished = {Software},
  url          = {https://alphafold.ebi.ac.uk/},
  doi          = {10.5281/zenodo.5123456}
}
```

## 擷取工作流程

### 從 DOI

**最佳實務** - 最可靠的來源：

```bash
# 單一 DOI
python scripts/extract_metadata.py --doi 10.1038/s41586-021-03819-2

# 多個 DOI
python scripts/extract_metadata.py \
  --doi 10.1038/nature12345 \
  --doi 10.1126/science.abc1234 \
  --output refs.bib
```

**流程**：
1. 使用 DOI 查詢 CrossRef API
2. 解析 JSON 回應
3. 擷取必填欄位
4. 決定條目類型（@article、@book 等）
5. 格式化為 BibTeX
6. 驗證完整性

### 從 PMID

**用於生物醫學文獻**：

```bash
# 單一 PMID
python scripts/extract_metadata.py --pmid 34265844

# 多個 PMID
python scripts/extract_metadata.py \
  --pmid 34265844 \
  --pmid 28445112 \
  --output refs.bib
```

**流程**：
1. 使用 PMID 查詢 PubMed EFetch
2. 解析 XML 回應
3. 擷取後設資料包括 MeSH 詞彙
4. 檢查回應中是否有 DOI
5. 如有 DOI，可選擇性查詢 CrossRef 取得額外後設資料
6. 格式化為 BibTeX

### 從 arXiv ID

**用於預印本**：

```bash
python scripts/extract_metadata.py --arxiv 2103.14030
```

**流程**：
1. 使用 ID 查詢 arXiv API
2. 解析 Atom XML 回應
3. 檢查是否有已發表版本（回應中的 DOI）
4. 如已發表：使用 DOI 和 CrossRef
5. 如未發表：使用預印本後設資料
6. 格式化為帶預印本註釋的 @misc

**重要**：始終檢查預印本是否已發表！

### 從 URL

**當您只有 URL 時**：

```bash
python scripts/extract_metadata.py \
  --url "https://www.nature.com/articles/s41586-021-03819-2"
```

**流程**：
1. 解析 URL 以擷取識別碼
2. 識別類型（DOI、PMID、arXiv）
3. 從 URL 擷取識別碼
4. 查詢適當的 API
5. 格式化為 BibTeX

**URL 模式**：
```
# DOI URL
https://doi.org/10.1038/nature12345
https://dx.doi.org/10.1126/science.abc123
https://www.nature.com/articles/s41586-021-03819-2

# PubMed URL
https://pubmed.ncbi.nlm.nih.gov/34265844/
https://www.ncbi.nlm.nih.gov/pubmed/34265844

# arXiv URL
https://arxiv.org/abs/2103.14030
https://arxiv.org/pdf/2103.14030.pdf
```

### 批次處理

**從包含混合識別碼的檔案**：

```bash
# 建立每行一個識別碼的檔案
# identifiers.txt：
#   10.1038/nature12345
#   34265844
#   2103.14030
#   https://doi.org/10.1126/science.abc123

python scripts/extract_metadata.py \
  --input identifiers.txt \
  --output references.bib
```

**流程**：
- 腳本自動檢測識別碼類型
- 查詢適當的 API
- 全部合併到單一 BibTeX 檔案
- 優雅地處理錯誤

## 特殊情況和邊緣案例

### 後來發表的預印本

**問題**：引用了預印本，但期刊版本現已可用。

**解決方案**：
1. 檢查 arXiv 後設資料中的 DOI 欄位
2. 如有 DOI，使用已發表版本
3. 將引用更新為期刊文章
4. 如需要，在註釋中註明預印本版本

**範例**：
```bibtex
% 原先：arXiv:2103.14030
% 已發表為：
@article{Jumper2021,
  author  = {Jumper, John and Evans, Richard and others},
  title   = {Highly Accurate Protein Structure Prediction with {AlphaFold}},
  journal = {Nature},
  year    = {2021},
  volume  = {596},
  pages   = {583--589},
  doi     = {10.1038/s41586-021-03819-2}
}
```

### 多位作者（et al.）

**問題**：許多作者（10 位以上）。

**BibTeX 實務**：
- 如少於 10 位則包含所有
- 10 位以上使用「and others」
- 或列出所有（期刊要求不同）

**範例**：
```bibtex
@article{LargeCollaboration2024,
  author = {First, Author and Second, Author and Third, Author and others},
  ...
}
```

### 作者姓名變體

**問題**：作者以不同姓名格式發表。

**標準化**：
```
# 常見變體
John Smith
John A. Smith
John Andrew Smith
J. A. Smith
Smith, J.
Smith, J. A.

# BibTeX 格式（建議）
author = {Smith, John A.}
```

**擷取優先順序**：
1. 如可用則使用全名
2. 如可用則包含中間名縮寫
3. 格式：Last, First Middle

### 無可用 DOI

**問題**：較舊的論文或書籍無 DOI。

**解決方案**：
1. 如可用則使用 PMID（生物醫學）
2. 書籍使用 ISBN
3. 使用穩定來源的 URL
4. 包含完整出版詳細資訊

**範例**：
```bibtex
@article{OldPaper1995,
  author  = {Author, Name},
  title   = {Title Here},
  journal = {Journal Name},
  year    = {1995},
  volume  = {123},
  pages   = {45--67},
  url     = {https://stable-url-here},
  note    = {PMID: 12345678}
}
```

### 會議論文 vs 期刊文章

**問題**：同一研究在兩處發表。

**最佳實務**：
- 如兩者都可用則引用期刊版本
- 期刊版本是存檔版
- 會議版本用於時效性

**如引用會議**：
```bibtex
@inproceedings{Smith2024conf,
  author    = {Smith, John},
  title     = {Title},
  booktitle = {Proceedings of NeurIPS 2024},
  year      = {2024}
}
```

**如引用期刊**：
```bibtex
@article{Smith2024journal,
  author  = {Smith, John},
  title   = {Title},
  journal = {Journal of Machine Learning Research},
  year    = {2024}
}
```

### 書籍章節 vs 編輯文集

**正確擷取**：
- 章節：使用 `@incollection`
- 整本書：使用 `@book`
- 書籍編輯：列在 `editor` 欄位
- 章節作者：列在 `author` 欄位

### 資料集和軟體

**使用 @misc** 並附上適當欄位：

```bibtex
@misc{DatasetName2024,
  author       = {Author, Name},
  title        = {Dataset Title},
  year         = {2024},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.123456},
  note         = {Version 1.2}
}
```

## 擷取後驗證

始終驗證擷取的後設資料：

```bash
python scripts/validate_citations.py extracted_refs.bib
```

**檢查**：
- 所有必填欄位存在
- DOI 正確解析
- 作者姓名格式一致
- 年份合理（4 位數）
- 期刊/出版商名稱正確
- 頁碼範圍使用 -- 而非 -
- 特殊字元處理正確

## 最佳實務

### 1. 可用時優先使用 DOI

DOI 提供：
- 永久識別碼
- 最佳後設資料來源
- 出版商驗證的資訊
- 可解析的連結

### 2. 驗證自動擷取的後設資料

抽查：
- 作者姓名與出版物相符
- 標題相符（包括大小寫）
- 年份正確
- 期刊名稱完整

### 3. 處理特殊字元

**LaTeX 特殊字元**：
- 保護大小寫：`{AlphaFold}`
- 處理重音：`M{\"u}ller` 或使用 Unicode
- 化學公式：`H$_2$O` 或 `\ce{H2O}`

### 4. 使用一致的引用鍵

**慣例**：`FirstAuthorYEARkeyword`
```
Smith2024protein
Doe2023machine
Johnson2024cancer
```

### 5. 現代論文包含 DOI

約 2000 年以後發表的所有論文都應有 DOI：
```bibtex
doi = {10.1038/nature12345}
```

### 6. 記錄來源

對於非標準來源，添加註釋：
```bibtex
note = {Preprint, not peer-reviewed}
note = {Technical report}
note = {Dataset accompanying [citation]}
```

## 總結

後設資料擷取工作流程：

1. **識別**：確定識別碼類型（DOI、PMID、arXiv、URL）
2. **查詢**：使用適當的 API（CrossRef、PubMed、arXiv）
3. **擷取**：解析回應取得必填欄位
4. **格式化**：建立正確格式的 BibTeX 條目
5. **驗證**：檢查完整性和準確性
6. **核實**：抽查關鍵引用

**使用腳本**自動化：
- `extract_metadata.py`：通用擷取器
- `doi_to_bibtex.py`：快速 DOI 轉換
- `validate_citations.py`：驗證準確性

**最終提交前務必驗證**擷取的後設資料！
